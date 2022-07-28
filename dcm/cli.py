"""Command line interface"""
from __future__ import annotations
import sys, os, logging, json, re, signal
import asyncio
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
from typing import Optional, Callable, Awaitable

import pydicom
from pydicom.dataset import Dataset
from pydicom.datadict import keyword_for_tag
from pynetdicom import evt
import click
import toml
from rich.console import Console
from rich.progress import Progress
from rich.logging import RichHandler
import dateparser

from . import __version__
from .conf import DcmConfig, _default_conf, NoLocalNodeError
from .util import str_to_tag, aclosing, json_serializer
from .lazyset import AllElems, LazySet
from .report import MultiListReport, RichProgressHook
from .net import (
    DcmNode,
    FailedAssociationError,
    LocalEntity,
    QueryLevel,
    EventFilter,
    make_queue_data_cb,
)
from .filt import make_edit_filter, MultiFilter
from .route import StaticRoute, DynamicTransferReport, Router
from .store.base import TransferMethod
from .store.local_dir import LocalDir
from .store.net_repo import NetRepo
from .sync import SyncReport, make_basic_validator, sync_data
from .normalize import normalize
from .diff import diff_data_sets


log = logging.getLogger("dcm.cli")


# Make sure we get SIGINT regardless of parent process mask
signal.signal(signal.SIGINT, signal.default_int_handler)


def cli_error(msg, exit_code=1):
    """Print msg to stderr and exit with non-zero exit code"""
    click.secho(msg, err=True, fg="red")
    sys.exit(exit_code)


class QueryResponseFilter(logging.Filter):
    def filter(self, record):
        if (
            record.name == "dcm.net"
            and record.levelno == logging.DEBUG
            and record.msg.startswith("Got query response:")
        ):
            return False
        return True


class PerformedQueryFilter(logging.Filter):
    def filter(self, record):
        if (
            record.name == "dcm.net"
            and record.levelno == logging.DEBUG
            and record.msg.startswith("Performing query:")
        ):
            return False
        return True


debug_filters = {
    "query_responses": QueryResponseFilter(),
    "performed_queries": PerformedQueryFilter(),
}


@click.group()
@click.option(
    "--config",
    type=click.Path(dir_okay=False, readable=True, resolve_path=True),
    envvar="DCM_CONFIG_PATH",
    default=os.path.join(click.get_app_dir("dcm"), "dcm_conf.toml"),
    help="Path to TOML config file",
)
@click.option(
    "--log-path",
    type=click.Path(dir_okay=False, readable=True, writable=True, resolve_path=True),
    envvar="DCM_LOG_PATH",
    help="Save logging output to this file",
)
@click.option(
    "--file-log-level",
    type=click.Choice(["DEBUG", "INFO", "WARN", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Log level to use when logging to a file",
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Print INFO log messages"
)
@click.option("--debug", is_flag=True, default=False, help="Print DEBUG log messages")
@click.option(
    "--debug-filter", multiple=True, help="Selectively filter debug log messages"
)
@click.option(
    "--quiet", is_flag=True, default=False, help="Hide WARNING and below log messages"
)
@click.option(
    "--pynetdicom-log-level",
    type=click.Choice(["DEBUG", "INFO", "WARN", "ERROR"], case_sensitive=False),
    default="WARN",
    help="Control log level for lower level pynetdicom package",
)
@click.pass_context
def cli(
    ctx,
    config,
    log_path,
    file_log_level,
    verbose,
    debug,
    debug_filter,
    quiet,
    pynetdicom_log_level,
):
    """High level DICOM file and network operations"""
    if quiet:
        if verbose or debug:
            cli_error("Can't mix --quiet with --verbose/--debug")

    # Create Rich Console outputing to stderr for logging / progress bars
    rich_con = Console(stderr=True)

    # Setup logging
    LOG_FORMAT = "%(asctime)s %(levelname)s %(threadName)s %(name)s %(message)s"
    def_formatter = logging.Formatter(LOG_FORMAT)
    root_logger = logging.getLogger("")
    root_logger.setLevel(logging.DEBUG)
    pynetdicom_logger = logging.getLogger("pynetdicom")
    pynetdicom_logger.setLevel(getattr(logging, pynetdicom_log_level))
    stream_formatter = logging.Formatter("%(threadName)s %(name)s %(message)s")
    stream_handler = RichHandler(console=rich_con, enable_link_path=False)
    stream_handler.setFormatter(stream_formatter)
    # logging.getLogger("asyncio").setLevel(logging.DEBUG)

    if debug:
        stream_handler.setLevel(logging.DEBUG)
    elif verbose:
        stream_handler.setLevel(logging.INFO)
    elif quiet:
        stream_handler.setLevel(logging.ERROR)
    else:
        stream_handler.setLevel(logging.WARN)
    root_logger.addHandler(stream_handler)
    handlers = [stream_handler]
    if log_path is not None:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(def_formatter)
        file_handler.setLevel(getattr(logging, file_log_level))
        root_logger.addHandler(file_handler)
        handlers.append(file_handler)

    if len(debug_filter) > 0:
        for filter_name in debug_filter:
            if filter_name not in debug_filters:
                cli_error("Unknown debug filter: %s" % filter_name)
            for handler in handlers:
                handler.addFilter(debug_filters[filter_name])

    # Create global param dict for subcommands to use
    ctx.obj = {}
    ctx.obj["config_path"] = config
    ctx.obj["config"] = DcmConfig(config, create_if_missing=True)
    ctx.obj["rich_con"] = rich_con


@click.command()
@click.pass_obj
def version(params):
    """Print the version and exit"""
    click.echo(__version__)


@click.command()
@click.pass_obj
@click.option("--show", is_flag=True, help="Just print the current config contents")
@click.option(
    "--show-default", is_flag=True, help="Just print the default config contets"
)
@click.option("--path", is_flag=True, help="Just print the current config path")
def conf(params, show, show_default, path):
    """Open the config file with your $EDITOR"""
    config_path = params["config_path"]
    # TODO: Make these mutually exclusive? Or sub-commands?
    if path:
        click.echo(config_path)
    if show:
        with open(config_path, "r") as f:
            click.echo(f.read())
    if show_default:
        click.echo(_default_conf)
    if path or show or show_default:
        return
    err = False
    while True:
        click.edit(filename=config_path)
        try:
            with open(config_path, "r") as f:
                _ = toml.load(f)
        except toml.decoder.TomlDecodeError as e:
            err = True
            click.echo("The config file contains an error: %s" % e)
            click.echo("The editor will be reopened so you can correct the error")
            click.pause()
        else:
            if err:
                click.echo("Config file is now valid")
            break


@click.command()
@click.pass_obj
@click.argument("remote")
@click.option("--local", help="Local DICOM network node properties")
def echo(params, remote, local):
    """Test connectivity with remote node"""
    local = params["config"].get_local_node(local)
    remote_node = params["config"].get_remote_node(remote)
    net_ent = LocalEntity(local)
    try:
        res = asyncio.run(net_ent.echo(remote_node))
    except FailedAssociationError:
        res = False
    if res:
        click.echo("Success")
    else:
        cli_error("Failed")


def _hr_to_dcm_date(in_str):
    try:
        dt = dateparser.parse(in_str)
    except Exception as e:
        cli_error(f"Unable to parse date '{in_str}': {e}")
    return dt.strftime("%Y%m%d")


def _build_study_date(since, before):
    if since is not None:
        since_str = _hr_to_dcm_date(since)
    else:
        since_str = ""
    if before is not None:
        before_str = _hr_to_dcm_date(before)
    else:
        before_str = ""
    return f"{since_str}-{before_str}"


def _build_query(query_strs, since, before):
    qdat = Dataset()
    for query_input in query_strs:
        try:
            q_attr, q_val = query_input.split("=")
        except Exception:
            cli_error(f"Invalid query input string: {query_input}")
        setattr(qdat, q_attr, q_val)
    if since is not None or before is not None:
        if hasattr(qdat, "StudyDate"):
            cli_error("Do not specify 'StudyDate' when using '--since' or '--before'")
        setattr(qdat, "StudyDate", _build_study_date(since, before))
    return qdat


@click.command()
@click.pass_obj
@click.argument("remote")
@click.argument("query", nargs=-1)
@click.option(
    "--level", default=None, help="Level of detail: patient/study/series/image"
)
@click.option(
    "--query-res",
    type=click.File("rb"),
    help="A result from a previous query to refine",
)
@click.option("--since", help="Only return studies since this date")
@click.option("--before", help="Only return studies before this date")
@click.option("--local", help="Local DICOM network node properties")
@click.option("--out-format", default=None, help="Output format: tree/json")
@click.option(
    "--assume-yes",
    is_flag=True,
    default=False,
    help="Automatically answer all prompts with 'y'",
)
@click.option("--no-progress", is_flag=True, help="Don't display progress bars")
def query(
    params,
    remote,
    query,
    level,
    query_res,
    since,
    before,
    local,
    out_format,
    assume_yes,
    no_progress,
):
    """Perform a query against a network node"""
    if level is not None:
        level = level.upper()
        for q_lvl in QueryLevel:
            if q_lvl.name == level:
                level = q_lvl
                break
        else:
            cli_error("Invalid level: %s" % level)
    if query_res is None and not sys.stdin.isatty():
        log.debug("Reading query_res from stdin")
        query_res = sys.stdin
    if query_res is not None:
        in_str = query_res.read()
        if in_str:
            query_res = json_serializer.loads(in_str)
        else:
            query_res = None

    if sys.stdout.isatty():
        if out_format is None:
            out_format = "tree"
    else:
        no_progress = True
        if out_format is None:
            out_format = "json"
    if out_format not in ("tree", "json"):
        cli_error("Invalid out-format: %s" % out_format)
    local = params["config"].get_local_node(local)
    remote_node = params["config"].get_remote_node(remote)
    net_ent = LocalEntity(local)
    qdat = _build_query(query, since, before)
    if len(qdat) == 0 and query_res is None and not assume_yes:
        if not click.confirm(
            "This query hasn't been limited in any "
            "way and may generate a huge result, "
            "continue?"
        ):
            return
    with ExitStack() as estack:
        if not no_progress:
            prog = RichProgressHook(
                estack.enter_context(
                    Progress(console=params["rich_con"], transient=True)
                )
            )
            report = MultiListReport(description="query", prog_hook=prog)
        else:
            report = MultiListReport(description="query")

        qr = asyncio.run(
            net_ent.query(remote_node, level, qdat, query_res, report=report)
        )
    if out_format == "tree":
        out = qr.to_tree()
    elif out_format == "json":
        out = json_serializer.dumps(qr, indent=4)
    click.echo(out)
    report.log_issues()


@click.command()
@click.pass_obj
@click.argument("dests", nargs=-1)
@click.option("--source", "-s", multiple=True, help="A data source")
@click.option("--query", "-q", multiple=True, help="Only sync data matching the query")
@click.option(
    "--query-res",
    type=click.File("rb"),
    help="A result from a previous query to limit the data synced",
)
@click.option("--since", help="Only return studies since this date")
@click.option("--before", help="Only return studies before this date")
@click.option(
    "--edit", "-e", multiple=True, help="Modify DICOM attribute in the synced data"
)
@click.option(
    "--edit-json", type=click.File("rb"), help="Specify attribute modifications as JSON"
)
@click.option(
    "--trust-level",
    type=click.Choice([q.name for q in QueryLevel], case_sensitive=False),
    default="IMAGE",
    help="If sub-component counts match at this query level, assume "
    "the data matches. Improves performance but sacrifices accuracy",
)
@click.option(
    "--force-all",
    "-f",
    is_flag=True,
    default=False,
    help="Force all data on the source to be transfered, even if it "
    "appears to already exist on the dest",
)
@click.option(
    "--method",
    "-m",
    help="Transfer method to use",
    type=click.Choice([m.name for m in TransferMethod], case_sensitive=False),
)
# Expose this when it is working
# @click.option('--validate', is_flag=True, default=False,
#              help="All synced data is retrieved back from the dests and "
#              "compared to the original data. Differing elements produce "
#              "warnings.")
@click.option(
    "--keep-errors",
    is_flag=True,
    default=False,
    help="Don't skip inconsistent/unexpected incoming data",
)
@click.option(
    "--dry-run",
    "-n",
    is_flag=True,
    default=False,
    help="Don't actually do any transfers, just print them",
)
@click.option("--local", help="Local DICOM network node properties")
@click.option("--dir-format", help="Output format for any local output directories")
@click.option(
    "--recurse/--no-recurse",
    default=None,
    is_flag=True,
    help="Don't recurse into input directories",
)
@click.option("--in-file-ext", help="File extension for local input directories")
@click.option("--out-file-ext", help="File extension for local output directories")
@click.option("--no-progress", is_flag=True, help="Don't display progress bars")
@click.option("--no-report", is_flag=True, help="Don't print report")
def sync(
    params,
    dests,
    source,
    query,
    query_res,
    since,
    before,
    edit,
    edit_json,
    trust_level,
    force_all,
    method,
    keep_errors,
    dry_run,
    local,
    dir_format,
    recurse,
    in_file_ext,
    out_file_ext,
    no_progress,
    no_report,
):
    """Sync DICOM data from a one or more sources to one or more destinations

    The `dests` can be a local directory, a DICOM network entity (given as
    'hostname:aetitle:port'), or a named remote/route from your config file.

    Generally you will need to use `--source` to specify the data source, unless
    you pass in a query result which contains a source (e.g. when doing
    'dcm query srcpacs ... | dcm sync destpacs'). The `--source` can be given
    in the same way `dests` are specified, except it cannot be a 'route'.
    """
    # Check for incompatible options
    # if validate and dry_run:
    #    cli_error("Can't do validation on a dry run!")

    # Disable progress for non-interactive output or dry runs
    if not sys.stdout.isatty() or dry_run:
        no_progress = True

    # Build query dataset if needed
    if len(query) != 0 or since is not None or before is not None:
        query = _build_query(query, since, before)
    else:
        query = None

    # Handle query-result options
    if query_res is None and not sys.stdin.isatty():
        query_res = sys.stdin
    if query_res is not None:
        in_str = query_res.read()
        if in_str:
            query_res = json_serializer.loads(in_str)
        else:
            query_res = None

    # Determine the local node being used
    try:
        local = params["config"].get_local_node(local)
    except NoLocalNodeError:
        local = None

    # Pass source options that override config through to the config parser
    local_dir_kwargs = {"make_missing": False}
    if recurse is not None:
        local_dir_kwargs["recurse"] = recurse
    if in_file_ext is not None:
        local_dir_kwargs["file_ext"] = in_file_ext
    params["config"].set_local_dir_kwargs(**local_dir_kwargs)
    params["config"].set_net_repo_kwargs(local=local)

    # Figure out source info
    if len(source) == 0:
        if query_res is None or query_res.prov.source is None:
            cli_error("No data source specified")
        if local is None:
            raise NoLocalNodeError("No local DICOM node configured")
        sources = [NetRepo(local, query_res.prov.source)]
    else:
        sources = []
        for s in source:
            try:
                sources.append(params["config"].get_bucket(s))
            except Exception as e:
                cli_error(f"Error processing source '{s}': {e}")

    # Pass dest options that override config through to the config parser
    local_dir_kwargs = {}
    if dir_format is not None:
        local_dir_kwargs["out_fmt"] = dir_format
    if out_file_ext is not None:
        local_dir_kwargs["file_ext"] = out_file_ext
    params["config"].set_local_dir_kwargs(**local_dir_kwargs)

    # Some command line options override route configuration
    static_route_kwargs = {}
    dynamic_route_kwargs = {}

    # Handle edit options
    filt = None
    if edit_json is not None:
        edit_dict = json.load(edit_json)
        edit_json.close()
    else:
        edit_dict = {}
    if edit:
        for edit_str in edit:
            attr, val = edit_str.split("=")
            edit_dict[attr] = val
    if edit_dict:
        filt = make_edit_filter(edit_dict)
        static_route_kwargs["filt"] = filt
        dynamic_route_kwargs["filt"] = filt

    # Convert dests/filters to a StaticRoute
    if method is not None:
        method = TransferMethod[method.upper()]
        static_route_kwargs["methods"] = (method,)
        dynamic_route_kwargs["methods"] = {None: (method,)}

    # Pass route options that override config through to the config parser
    params["config"].set_static_route_kwargs(**static_route_kwargs)
    params["config"].set_dynamic_route_kwargs(**dynamic_route_kwargs)

    # Do samity check that no sources are in dests. This is especially easy
    # mistake as earlier versions took the first positional arg to be the
    # source
    for dest in dests:
        try:
            d_bucket = params["config"].get_bucket(dest)
        except Exception:
            pass
        else:
            if any(s == d_bucket for s in sources):
                cli_error(f"The dest {dest} is also a source!")
            continue
        try:
            static_route = params["config"].get_static_route(dest)
        except Exception:
            pass
        else:
            for d in static_route.dests:
                if any(s == d_bucket for s in sources):
                    cli_error(f"The dest {d} is also a source!")
            continue
        try:
            sel_dest_map = params["config"].get_selector_dest_map(dest)
        except Exception:
            pass
        else:
            for _, s_dests in sel_dest_map.routing_map:
                for d in s_dests:
                    if any(s == d_bucket for s in sources):
                        cli_error(f"The dest {d} is also a source!")
            continue
        cli_error(f"Unknown dest: {dest}")

    # Convert dests to routes
    dests = params["config"].get_routes(dests)

    # Handle validate option
    # if validate:
    #    validators = [make_basic_validator()]
    # else:
    #    validators = None

    # Handle trust-level option
    trust_level = QueryLevel[trust_level.upper()]

    # Setup reporting/progress hooks and do the transfer
    with ExitStack() as estack:
        if not no_progress:
            prog_hook = RichProgressHook(
                estack.enter_context(
                    Progress(console=params["rich_con"], transient=True)
                )
            )

        qr_reports = None
        if query is not None or query_res is not None:
            qr_reports = []
            for src in sources:
                if not no_progress:
                    report = MultiListReport(
                        description="init-query", prog_hook=prog_hook
                    )
                else:
                    report = None
                qr_reports.append(report)

        if query_res is None:
            qrs = None
        else:
            qrs = [deepcopy(query_res) for _ in sources]

        base_kwargs = {
            "trust_level": trust_level,
            "force_all": force_all,
            "keep_errors": keep_errors,
            #'validators': validators,
        }
        sm_kwargs = []
        sync_reports = []
        for src in sources:
            if not no_progress:
                sync_report = SyncReport(prog_hook=prog_hook)
            else:
                sync_report = SyncReport()
            kwargs = deepcopy(base_kwargs)
            kwargs["report"] = sync_report
            sm_kwargs.append(kwargs)
            sync_reports.append(sync_report)

        asyncio.run(
            sync_data(sources, dests, query, qrs, qr_reports, sm_kwargs, dry_run)
        )

    for report in sync_reports:
        report.log_issues()
        if not no_report:
            click.echo(report)
            click.echo("\n")


def _make_route_data_cb(
    res_q: asyncio.Queue[Dataset],
) -> Callable[[evt.Event], Awaitable[int]]:
    """Return callback that queues dataset/metadata from incoming events"""

    async def callback(event: evt.Event) -> int:
        # TODO: Do we need to embed the file_meta here?
        await res_q.put(event.dataset)
        return 0x0  # Success

    return callback


async def _do_route(
    local: DcmNode, router: Router, inactive_timeout: Optional[int] = None
) -> None:
    local_ent = LocalEntity(local)
    event_filter = EventFilter(event_types=frozenset((evt.EVT_C_STORE,)))
    report = DynamicTransferReport()
    last_update = None
    if inactive_timeout:
        last_update = datetime.now()
        last_reported = 0
    async with router.route(report=report) as route_q:
        fwd_cb = _make_route_data_cb(route_q)
        async with local_ent.listen(fwd_cb, event_filter=event_filter):
            print("Listener started, hit Ctrl-c to exit")
            try:
                while True:
                    await asyncio.sleep(1.0)
                    if last_update is not None:
                        n_reported = report.n_reported
                        if n_reported != last_reported:
                            last_update = datetime.now()
                            last_reported = n_reported
                        elif inactive_timeout is not None:
                            if (
                                datetime.now() - last_update
                            ).total_seconds() > inactive_timeout:
                                print("Timeout due to inactivity")
                                break
            finally:
                print("Listener shutting down")


@click.command()
@click.pass_obj
@click.argument("dests", nargs=-1)
@click.option(
    "--edit", "-e", multiple=True, help="Modify DICOM attribute in the synced data"
)
@click.option(
    "--edit-json", type=click.File("rb"), help="Specify attribute modifications as JSON"
)
@click.option("--local", help="Local DICOM network node properties")
@click.option("--dir-format", help="Output format for any local output directories")
@click.option(
    "--out-file-ext", default="dcm", help="File extension for local output directories"
)
@click.option(
    "--inactive-timeout",
    type=int,
    help="Stop listening after this many seconds of inactivity",
)
def forward(
    params, dests, edit, edit_json, local, dir_format, out_file_ext, inactive_timeout
):
    """Listen for incoming DICOM files on network and forward to dests"""
    local = params["config"].get_local_node(local)

    # Pass dest options that override config through to the config parser
    params["config"].set_local_dir_kwargs(out_fmt=dir_format, file_ext=out_file_ext)

    # Some command line options override route configuration
    static_route_kwargs = {}
    dynamic_route_kwargs = {}

    # Handle edit options
    filt = None
    if edit_json is not None:
        edit_dict = json.load(edit_json)
        edit_json.close()
    else:
        edit_dict = {}
    if edit:
        for edit_str in edit:
            attr, val = edit_str.split("=")
            edit_dict[attr] = val
    if edit_dict:
        filt = make_edit_filter(edit_dict)
        static_route_kwargs["filt"] = filt
        dynamic_route_kwargs["filt"] = filt

    # Pass route options that override config through to the config parser
    params["config"].set_static_route_kwargs(**static_route_kwargs)
    params["config"].set_dynamic_route_kwargs(**dynamic_route_kwargs)

    # Convert dests to routes
    dests = params["config"].get_routes(dests)

    router = Router(dests)
    asyncio.run(_do_route(local, router, inactive_timeout))


def make_print_cb(fmt, elem_filter=None):
    def print_cb(ds, elem):
        if elem_filter:
            tag = elem.tag
            keyword = keyword_for_tag(tag)
            if not elem_filter(tag, keyword):
                return
        try:
            print(fmt.format(elem=elem, ds=ds))
        except Exception:
            log.warn("Couldn't apply format to elem: %s", elem)

    return print_cb


def _make_elem_filter(include, exclude, groups, kw_regex, exclude_private):
    if len(include) == 0:
        include_tags = LazySet(AllElems)
    else:
        include_tags = set()
        for in_str in include:
            include_tags.add(str_to_tag(in_str))
        include_tags = LazySet(include_tags)
    exclude_tags = set()
    for in_str in exclude:
        exclude_tags.add(str_to_tag(in_str))
    exclude_tags = LazySet(exclude_tags)
    if len(groups) == 0:
        groups = LazySet(AllElems)
    else:
        groups = LazySet([int(x) for x in groups])
    kw_regex = [re.compile(x) for x in kw_regex]

    def elem_filter(tag, keyword):
        if exclude_private and tag.group % 2 == 1:
            return False
        if tag in exclude_tags:
            return False
        if tag.group not in groups:
            if include and tag in include_tags:
                return True
            return False
        if kw_regex:
            keyword = keyword_for_tag(tag)
            if not any(r.search(keyword) for r in kw_regex):
                if include and tag in include_tags:
                    return True
                return False
        if tag in include_tags:
            return True
        return False

    return elem_filter


@click.command()
@click.pass_obj
@click.argument("dcm_files", type=click.Path(exists=True, readable=True), nargs=-1)
@click.option("--out-format", default="plain", help="Output format: plain/json")
@click.option(
    "--plain-fmt",
    default="{elem}",
    help="Format string applied to each element for 'plain' output. Can "
    "reference 'elem' (the pydicom Element) and 'ds' (the pydicom Dataset) "
    "objects in the format string.",
)
@click.option(
    "--include",
    "-i",
    multiple=True,
    help="Include specific elements by keyword or tag",
)
@click.option("--group", "-g", multiple=True, help="Include elements by group number")
@click.option(
    "--kw-regex",
    multiple=True,
    help="Include elements where the keyword matches a regex",
)
@click.option(
    "--exclude", "-e", multiple=True, help="Exclude elements by keyword or tag"
)
@click.option(
    "--exclude-private",
    is_flag=True,
    default=False,
    help="Exclude all private elements",
)
def dump(
    params,
    dcm_files,
    out_format,
    plain_fmt,
    include,
    group,
    kw_regex,
    exclude,
    exclude_private,
):
    """Dump contents of DICOM files to stdout

    Default is to include all elements, but this can be overridden by various options.

    The `--out-format` can be `plain` or `json`. If it is `plain` each included element
    is used to format a string which can be overridden with `--plain-fmt`.
    """
    elem_filter = _make_elem_filter(include, exclude, group, kw_regex, exclude_private)
    if out_format == "plain":
        print_cb = make_print_cb(plain_fmt, elem_filter)
        for pth in dcm_files:
            ds = pydicom.dcmread(pth)
            ds.walk(print_cb)
    elif out_format == "json":
        for pth in dcm_files:
            ds = pydicom.dcmread(pth)
            click.echo(json.dumps(normalize(ds, elem_filter), indent=4))
    else:
        cli_error("Unknown out-format: '%s'" % out_format)


@click.command()
@click.pass_obj
@click.argument("left")
@click.argument("right")
def diff(params, left, right):
    """Show differences between two data sets"""
    left = pydicom.dcmread(left)
    right = pydicom.dcmread(right)
    diffs = diff_data_sets(left, right)
    for d in diffs:
        click.echo(str(d))


# Add our subcommands ot the CLI
cli.add_command(version)
cli.add_command(conf)
cli.add_command(echo)
cli.add_command(query)
cli.add_command(sync)
cli.add_command(forward)
cli.add_command(dump)
cli.add_command(diff)


# Entry point
if __name__ == "__main__":
    cli()

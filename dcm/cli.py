'''Command line interface'''
import sys, os, logging, json, textwrap
import asyncio

import pydicom
from pydicom.dataset import Dataset

import click

import toml

from .util import aclosing
from .query import QueryResult
from .net import DcmNode, LocalEntity, QueryLevel
from .filt import make_edit_filter
from .route import StaticRoute
from .store.local_dir import LocalDir
from .store.net_repo import NetRepo
from .sync import TransferPlanner, make_basic_validator
from .normalize import normalize
from .diff import diff_data_sets


log = logging.getLogger('dcm.cli')

#logging.basicConfig(level=logging.DEBUG)
logging.getLogger("asyncio").setLevel(logging.DEBUG)


def cli_error(msg, exit_code=1):
    '''Print msg to stderr and exit with non-zero exit code'''
    click.secho(msg, err=True, fg='red')
    sys.exit(exit_code)


def parse_node_spec(in_str, conf_nodes=None):
    '''Parse a string into a DcmNode (host/ae_title/port)'''
    if conf_nodes is not None and in_str in conf_nodes:
        return conf_nodes[in_str]
    toks = in_str.split(':')
    if len(toks) > 3:
        raise ValueError("Too many tokens for node specification: %s" %
                         in_str)
    host = toks[0]
    if len(toks) == 3:
        ae_title, port = toks[1], int(toks[2])
    elif len(toks) == 2:
        try:
            port = int(toks[1])
        except ValueError:
            port = 104
            ae_title = toks[1]
        else:
            ae_title = 'ANYAE'
    elif len(toks) == 1:
        ae_title = 'ANYAE'
        port = 104
    return DcmNode(host, ae_title, port)


def parse_target(in_str, local_node=None, conf_nodes=None, out_fmt=None,
                 no_recurse=False, file_ext='dcm'):
    '''Parse command line argument into a target for data transfers'''
    if conf_nodes is not None and in_str in conf_nodes:
        remote = conf_nodes[in_str]
        return NetRepo(local_node, remote)
    if os.path.isdir(in_str):
        return LocalDir(in_str, out_fmt=out_fmt, recurse=not no_recurse,
                        file_ext=file_ext)
    remote = parse_node_spec(in_str, conf_nodes)
    return NetRepo(local_node, remote)


def node_from_conf(conf_dict, default_host=None, default_ae='ANYAE',
                   default_port=104):
    res = DcmNode(conf_dict.get('host', default_host),
                  conf_dict.get('ae_title', default_ae),
                  conf_dict.get('port', default_port),
                 )
    if res.host is None:
        raise ValueError("A hostname is required")
    return res


_default_conf = \
'''
# Uncomment and add your local AE_Title / port
#[local]
#ae_title = "YOURAE"
#port = 11112

# Uncomment and add any PACS or other DICOM network entities
#[remotes]
#
#  [remotes.yourpacs]
#  host = "yourpacs.example.org"
#  ae_title = "PACSAETITLE"
#  port = 104

'''

class QueryResponseFilter(logging.Filter):
    def filter(self, record):
        if (record.name == 'dcm.net' and
            record.levelno == logging.DEBUG and
            record.msg.startswith("Got query response:")
           ):
            return False
        return True


class PerformedQueryFilter(logging.Filter):
    def filter(self, record):
        if (record.name == 'dcm.net' and
            record.levelno == logging.DEBUG and
            record.msg.startswith("Performing query:")
           ):
            return False
        return True

debug_filters = {'query_responses' : QueryResponseFilter(),
                 'performed_queries' : PerformedQueryFilter(),
                }


@click.group()
@click.option('--config',
              type=click.Path(dir_okay=False,
                              readable=True,
                              resolve_path=True),
              envvar='DCM_CONFIG_PATH',
              default=os.path.join(click.get_app_dir('dcm'), 'dcm_conf.toml'),
              help="Path to TOML config file",
             )
@click.option('--log-path',
              type=click.Path(dir_okay=False,
                              readable=True,
                              writable=True,
                              resolve_path=True),
              envvar='DCM_LOG_PATH',
              help="Save logging output to this file")
@click.option('--verbose', '-v',
              is_flag=True,
              default=False,
              help="Print INFO log messages")
@click.option('--debug',
              is_flag=True,
              default=False,
              help="Print DEBUG log messages")
@click.option('--debug-filter',
              multiple=True,
              help="Selectively filter debug log messages")
@click.pass_context
def cli(ctx, config, log_path, verbose, debug, debug_filter):
    '''High level DICOM file and network operations
    '''
    # Parse the config file
    if os.path.exists(config):
        with open(config, 'r') as f:
            conf_dict = toml.load(f)
    else:
        config_dir = os.path.dirname(config)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        with open(config, 'w') as f:
            f.write(_default_conf)
        conf_dict = {}

    # Setup logging
    LOG_FORMAT = '%(asctime)s %(levelname)s %(threadName)s %(name)s %(message)s'
    formatter = logging.Formatter(LOG_FORMAT)
    root_logger = logging.getLogger('')
    root_logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    if debug:
        stream_handler.setLevel(logging.DEBUG)
    elif verbose:
        stream_handler.setLevel(logging.INFO)
    else:
        stream_handler.setLevel(logging.WARN)
    root_logger.addHandler(stream_handler)
    handlers = [stream_handler]
    if log_path is not None:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        handlers.append(file_handler)

    if len(debug_filter) > 0:
        for filter_name in debug_filter:
            if filter_name not in debug_filters:
                cli_error("Unknown debug filter: %s" % filter_name)
            for handler in handlers:
                handler.addFilter(debug_filters[filter_name])


    # Setup any DICOM nodes specified in the config file
    local_info = conf_dict.get('local', {})
    local_node = node_from_conf(local_info,
                                default_host='0.0.0.0',
                                default_port=11112)
    remote_nodes = {}
    node_specs = conf_dict.get('remotes', {})
    for node_name, node_spec in node_specs.items():
        remote_nodes[node_name] = node_from_conf(node_spec)

    # Create global param dict for subcommands to use
    ctx.obj = {}
    ctx.obj['config_path'] = config
    ctx.obj['config'] = conf_dict
    ctx.obj['local_node'] = local_node
    ctx.obj['remote_nodes'] = remote_nodes


@click.command()
@click.pass_obj
def conf(params):
    '''Open the config file with your $EDITOR'''
    config_path = params['config_path']
    while True:
        click.edit(filename=config_path)
        try:
            with open(config_path, 'r') as f:
                _ = toml.load(f)
        except toml.decoder.TomlDecodeError as e:
            click.echo("The config file contains an error: %s" % e)
            click.echo("The editor will be reopened so you can correct the error")
            click.pause()
        else:
            break


@click.command()
@click.pass_obj
@click.argument('remote')
@click.option('--local',
              help="Local DICOM network node properties")
def echo(params, remote, local):
    '''Test connectivity with remote node'''
    remote_node = parse_node_spec(remote, conf_nodes=params['remote_nodes'])
    if local is not None:
        local = parse_node_spec(local)
    else:
        local = params['local_node']
    net_ent = LocalEntity(local)
    res = asyncio.run(net_ent.echo(remote_node))
    if res:
        click.echo("Success")
    else:
        cli_error("Failed")


@click.command()
@click.pass_obj
@click.argument('remote')
@click.argument('query', nargs=-1)
@click.option('--level',
              default=None,
              help="Level of detail: patient/study/series/image")
@click.option('--query-res',
              type=click.File('rb'),
              help='A result from a previous query to refine')
@click.option('--local',
              help="Local DICOM network node properties")
@click.option('--out-format',
              default=None,
              help="Output format: tree/json")
@click.option('--assume-yes',
              is_flag=True,
              default=False,
              help="Automatically answer all prompts with 'y'")
def query(params, remote, query, level, query_res, local, out_format,
          assume_yes):
    '''Perform a query against a network node'''
    if level is not None:
        level = level.upper()
        for q_lvl in QueryLevel:
            if q_lvl.name == level:
                level = q_lvl
                break
        else:
            cli_error("Invalid level: %s" % level)
    if query_res is None and not sys.stdin.isatty():
        query_res = sys.stdin
    if query_res is not None:
        query_res = QueryResult.from_json(query_res.read())
    if out_format is None:
        if sys.stdout.isatty():
            out_format = 'tree'
        else:
            out_format = 'json'
    elif out_format not in ('tree', 'json'):
        cli_error("Invalid out-format: %s" % out_format)
    if len(query) == 0 and query_res is None and not assume_yes:
        if not click.confirm("This query hasn't been limited in any "
                             "way and may generate a huge result, "
                             "continue?"):
            return
    remote_node = parse_node_spec(remote, conf_nodes=params['remote_nodes'])
    if local is not None:
        local = parse_node_spec(local)
    else:
        local = params['local_node']
    net_ent = LocalEntity(local)
    qdat = Dataset()
    for query_input in query:
        q_attr, q_val = query_input.split('=')
        setattr(qdat, q_attr, q_val)
    qr = asyncio.run(net_ent.query(remote_node, level, qdat, query_res))
    if out_format == 'tree':
        out = qr.to_tree()
    elif out_format == 'json':
        out = qr.to_json()
    click.echo(out)


def _cancel_all_tasks(loop):
    to_cancel = asyncio.tasks.all_tasks(loop)
    if not to_cancel:
        return
    log.debug("Canceling tasks: %s" % to_cancel)
    for task in to_cancel:
        task.cancel()

    loop.run_until_complete(
        asyncio.tasks.gather(*to_cancel, loop=loop, return_exceptions=True))

    for task in to_cancel:
        if task.cancelled():
            continue
        if task.exception() is not None:
            log.debug("Got exception from cancelled task: %s" % task.exception())
            #import pdb ; pdb.set_trace()
            loop.call_exception_handler({
                'message': 'unhandled exception during asyncio.run() shutdown',
                'exception': task.exception(),
                'task': task,
            })



def aio_run(main, *, debug=False):
    """Run a coroutine.

    This function runs the passed coroutine, taking care of
    managing the asyncio event loop and finalizing asynchronous
    generators.

    This function cannot be called when another asyncio event loop is
    running in the same thread.

    If debug is True, the event loop will be run in debug mode.

    This function always creates a new event loop and closes it at the end.
    It should be used as a main entry point for asyncio programs, and should
    ideally only be called once.

    Example:

        async def main():
            await asyncio.sleep(1)
            print('hello')

        asyncio.run(main())
    """
    if asyncio.events._get_running_loop() is not None:
        raise RuntimeError(
            "asyncio.run() cannot be called from a running event loop")

    if not asyncio.coroutines.iscoroutine(main):
        raise ValueError("a coroutine was expected, got {!r}".format(main))

    loop = asyncio.events.new_event_loop()
    try:
        asyncio.events.set_event_loop(loop)
        loop.set_debug(debug)
        return loop.run_until_complete(main)
    finally:
        log.debug("Cleaning up in aio_run")
        log.debug("Found %d open asyncgens: %s" % (len(loop._asyncgens), loop._asyncgens))
        #import pdb ; pdb.set_trace()
        try:
            _cancel_all_tasks(loop)
            log.debug("Shutting down async generators")
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            log.debug("Closing loop")
            asyncio.events.set_event_loop(None)
            loop.close()


async def _do_sync(src, dests, query, query_res, dest_route, trust_level, force_all, dry_run, validators, proxy, keep_errors):
    # Perform initial query if needed
    if len(query) > 0:
        log.info("Querying source for initial data list")
        qdat = Dataset()
        for query_input in query:
            q_attr, q_val = query_input.split('=')
            setattr(qdat, q_attr, q_val)
        # TODO: Should we do this iteratively in the background too?
        #       The transfer planner would need to take a QR generator
        #       instead of a QR then.
        #
        #       Won't work if simultaneous associations limit is very low (e.g. 1) on
        #       the src, since we may need to perform further queries on src
        #       (series/image level) when determining what data is missing from
        #       the dests. Also, this should generally be somewhat quick, since
        #       it should be a low-level query.
        query_res = await src.query(query=qdat, query_res=query_res)

    # Setup transfer planner
    planner = TransferPlanner(src,
                              [dest_route],
                              trust_level=trust_level,
                              force_all=force_all)

    # Perform the sync or dry run
    log.info("Syncing data from '%s' to '%s'" %
             (src, ', '.join(str(x) for x in dests)))

    if dry_run:
        log.info("Starting dry run")
        async for transfer in planner.gen_transfers(query_res):
            print('%s > %s' % (transfer.chunk, transfer.routes))
        log.info("Finished dry run")
    else:
        log.info("Starting data sync")
        async with planner.executor(validators, proxy, keep_errors) as ex:
            report = ex.report
            async with aclosing(planner.gen_transfers(query_res)) as tgen:
                async for transfer in tgen:
                    await ex.exec_transfer(transfer)
        report.log_issues()
        report.check_errors()
        log.info("Finished data sync")



@click.command()
@click.pass_obj
@click.argument('src')
@click.argument('dests', nargs=-1)
@click.option('--proxy', '-p', is_flag=True, default=False,
              help="Retrieve data locally and then forward to destinations")
@click.option('--query', '-q', multiple=True,
              help="Only sync data matching the query")
@click.option('--query-res',
              type=click.File('rb'),
              help='A result from a previous query to limit the data synced')
@click.option('--edit', '-e', multiple=True,
              help="Modify DICOM attribute in the synced data")
@click.option('--edit-json', type=click.File('rb'),
              help="Specify attribute modifications as JSON")
@click.option('--trust-level',
              help="If sub-component counts match at this query level, assume "
              "the data matches. Improves performance but sacrifices accuracy")
@click.option('--force-all', '-f', is_flag=True, default=False,
              help="Force all data on src to be transfered, even if it "
              "appears to already exist on the dest")
@click.option('--validate', is_flag=True, default=False,
              help="All synced data is retrieved back from the dests and "
              "compared to the original data. Differing elements produce "
              "warnings.")
@click.option('--keep-errors', is_flag=True, default=False,
              help="Don't skip inconsistent/unexpected incoming data")
@click.option('--dry-run', '-n', is_flag=True, default=False,
              help="Don't actually do any transfers, just print them")
@click.option('--local',
              help="Local DICOM network node properties")
@click.option('--dir-format',
              help="Output format for any local output directories")
@click.option('--no-recurse', is_flag=True, default=False,
              help="Don't recurse into input directories")
@click.option('--file-ext', default='dcm',
              help="File extension for local input directories")
def sync(params, src, dests, proxy, query, query_res, edit, edit_json,
         trust_level, force_all, validate, keep_errors, dry_run, local,
         dir_format, no_recurse, file_ext):
    '''Synchronize DICOM data between network nodes and/or directories
    '''
    # Check for incompatible options
    if validate and dry_run:
        cli_error("Can't do validation on a dry run!")

    # Figure out any local/src/remote info
    if local is not None:
        local = parse_node_spec(local)
    else:
        local = params['local_node']
    src = parse_target(src,
                       local_node=local,
                       conf_nodes=params['remote_nodes'],
                       out_fmt=dir_format,
                       no_recurse=no_recurse,
                       file_ext=file_ext)
    dests = [parse_target(x,
                          local_node=local,
                          conf_nodes=params['remote_nodes'],
                          out_fmt=dir_format,
                          no_recurse=no_recurse,
                          file_ext=file_ext)
             for x in dests]

    # Handle query-result options
    if query_res is None and not sys.stdin.isatty():
        query_res = sys.stdin
    if query_res is not None:
        query_res = QueryResult.from_json(query_res.read())

    # Handle edit options
    filt = None
    if edit_json is not None:
        edit_dict = json.load(edit_json)
        edit_json.close()
    else:
        edit_dict = {}
    if edit:
        for edit_str in edit:
            attr, val = edit_str.split('=')
            edit_dict[attr] = val
    if edit_dict:
        filt = make_edit_filter(edit_dict)

    # Convert dests/filters to a StaticRoute
    dest_route = StaticRoute(dests, filt)

    # Handle validate option
    if validate:
        validators = [make_basic_validator()]
    else:
        validators = None

    # Handle trust-level option
    if trust_level is not None:
        trust_level = trust_level.upper()
        for q_lvl in QueryLevel:
            if q_lvl.name == trust_level:
                trust_level = q_lvl
                break
        else:
            cli_error("Invalid level: %s" % trust_level)

    aio_run(_do_sync(src, dests, query, query_res, dest_route, trust_level,
                     force_all, dry_run, validators, proxy, keep_errors))


async def _netdump_cb(event):
    #click.echo(event)
    #for attr in dir(event):
    #    click.echo(f'{attr} : {getattr(event)}')
    return 0x0


async def _netdump(net_ent):
    async with net_ent.listen(_netdump_cb):
        log.debug("Listening in _netdump")
        while True:
            await asyncio.sleep(0.1)


@click.command()
@click.pass_obj
@click.option('--local',
              help="Local DICOM network node properties")
def netdump(params, local):
    '''Listen for network events and print them out'''
    if local is not None:
        local = parse_node_spec(local)
    else:
        local = params['local_node']
    net_ent = LocalEntity(local)
    asyncio.run(_netdump(net_ent), debug=True)





@click.command()
@click.pass_obj
@click.argument('dests', nargs=-1)
@click.option('--edit', '-e', multiple=True,
              help="Modify DICOM attribute in the synced data")
@click.option('--edit-json', type=click.File('rb'),
              help="Specify attribute modifications as JSON")
@click.option('--local',
              help="Local DICOM network node properties")
@click.option('--dir-format',
              help="Output format for any local output directories")
def listen(params, dests, dir_format, edit, edit_json, local):
    '''Listen for incoming DICOM files on network and store in dest
    '''
    # Figure out any local/src/remote info
    if local is not None:
        local = parse_node_spec(local)
    else:
        local = params['local_node']
    dests = [parse_target(x,
                          local_node=local,
                          conf_nodes=params['remote_nodes'],
                          out_fmt=dir_format)
             for x in dests]
    # TODO: Need to further separate the logic in the sync module before we
    #       can easily implement this
    raise NotImplementedError


def make_print_cb(fmt, elem_filter=None):
    def print_cb(ds, elem):
        if elem_filter:
            elem = elem_filter(elem)
            if elem is None:
                return
        print(fmt.format(elem=elem))
    return print_cb


def make_ignore_filter(ignore_ids, ignore_private):
    ignore_keys = set()
    ignore_tags = set()
    for ig_id in ignore_ids:
        if ig_id[0].isupper():
            ignore_keys.add(ig_id)
        else:
            try:
                group_num, elem_num = [int(x, 0) for x in ig_id.split(',')]
            except Exception:
                raise ValueError("Invalid element ID: %s" % ig_id)
            ignore_tags.add(pydicom.tag.Tag(group_num, elem_num))
    def ignore_filter(elem):
        if elem.keyword in ignore_keys:
            return None
        if elem.tag in ignore_tags:
            return None
        if ignore_private and elem.tag.group % 2 == 1:
            return None
        return elem
    return ignore_filter


@click.command()
@click.pass_obj
@click.argument('dcm_files',
                type=click.Path(exists=True, readable=True),
                nargs=-1)
@click.option('--out-format',
              default='plain',
              help="Output format: plain/json")
@click.option('--plain-fmt',
              default='{elem}',
              help="Format string applied to each element for 'plain' output")
@click.option('--ignore', '-i', multiple=True,
              help='Ignore elements by keyword or tag')
@click.option('--ignore-private', is_flag=True, default=False,
              help="Ignore all private elements")
def dump(params, dcm_files, out_format, plain_fmt, ignore, ignore_private):
    '''Dump contents of DICOM files'''
    if ignore:
        ignore_filter = make_ignore_filter(ignore, ignore_private)
    else:
        ignore_filter = None
    if out_format == 'plain':
        print_cb = make_print_cb(plain_fmt, ignore_filter)
        for pth in dcm_files:
            ds = pydicom.dcmread(pth)
            ds.walk(print_cb)
    elif out_format == 'json':
        for pth in dcm_files:
            ds = pydicom.dcmread(pth)
            click.echo(json.dumps(normalize(ds, ignore_filter), indent=4))
    else:
        cli.error("Unknown out-format: '%s'" % out_format)


@click.command()
@click.pass_obj
@click.argument('left')
@click.argument('right')
def diff(params, left, right):
    '''Show differences between two data sets'''
    left = pydicom.dcmread(left)
    right = pydicom.dcmread(right)
    diffs = diff_data_sets(left, right)
    for d in diffs:
        click.echo(str(d))


# Add our subcommands ot the CLI
cli.add_command(conf)
cli.add_command(echo)
cli.add_command(query)
cli.add_command(sync)
#cli.add_command(listen)
cli.add_command(netdump)
cli.add_command(dump)
cli.add_command(diff)


# Entry point
if __name__ == '__main__':
    cli()

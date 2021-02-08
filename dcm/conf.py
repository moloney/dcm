# Config parsing
import os
from typing import Dict, List, Optional

import toml
import click

from .util import TomlConfigurable, InlineConfigurable
from .net import DcmNode, LocalEntity
from .store.local_dir import LocalDir
from .store.net_repo import NetRepo
from .filt import Selector, SingleSelector, MultiSelector, MultiFilter
from .route import StaticRoute, SelectorDestMap


class InvalidConfigError:
    '''Raised if invalid configuration is detected'''


_default_conf = \
'''
########################################################
## Basic Configuration
########################################################

## Uncomment and add your default local AE_Title / port.

#[local_nodes]
#
#  [local_nodes.default]
#  ae_title = "YOURAE"
#  port = 11112
#
#  # Usually you will just have the one 'default' one, but
#  # you can configure other local nodes if needed.
#  [local_nodes.other_local]
#  ae_title = "MYOTHERAE"
#  port = 11113


## Uncomment and add any PACS or other DICOM network entities

#[remote_nodes]
#
#  [remote_nodes.yourpacs]
#  host = "yourpacs.example.org"
#  ae_title = "PACSAETITLE"
#  port = 104


##############################################################
## Advanced Configuration
##############################################################

## Optionally add local directories w/ specialized parameters by
## defining 'local_dirs'. The example below modifies default 
## 'out_fmt' to append InstanceNumber to the filename

#[local_dirs]
#
#  [local_dirs.dicom_dir]
#  path = "~/dicom"
#  out_fmt = """
# {d.PatientID}/
# {d.StudyInstanceUID}/
# {d.Modality}/
# {d.SeriesNumber:03d}-{d.SeriesDescription}/
# {d.SOPInstanceUID}-{d.InstanceNumber:04d}"""


## Optionally configure remote DICOM repos beyond just the 
## AE/port/host by defining 'remote_repos'.

#[remote_repos]
#
#  [remote_repos.otherpacs]
#  local = "other_local"
#  remote.host = "otherpacs.example.org"
#  remote.ae_title = "OTHERPACS"


## Optionally define "selectors" that select data based on arbitrary 
## attributes, not just those the device supports querying on. 
## Selectors can also logically combine other selectors through the
## attributes "none_of", "all_of", and "any_of".

#[selectors]
#
#  [selectors.internal]
#  attr = "DeviceSerialNumber"
#  op = "in"
#  rvalue = [ "12345", "67890" ]
#
#  [selectors.external]
#  none_of = [ "internal" ]  


## Optionally specify routes for more control over how data is sent.
## These routes can be static or dynamic

#[static_routes]
#
#  # Route to two PACS with direct Move-SCU instead of proxying data
#  [routes.movescu_pacs]
#  dests = [ "yourpacs", "otherpacs" ]
#  methods = [ "REMOTE_COPY" ]
#
#  # Define simple selector inline and use it as a filter
#  [routes.filt_mypacs]
#  dests = [ "yourpacs" ]
#  filt = "StudyDescription == ProjectA"
#
#[dynamic_routes]
#
#  # We can specify a dynamic route by groups selectors with dests
#  [routes.dyn_route]
#  route_map = [ [ "internal", [ "yourpacs" ] ], 
#                [ "external", [ "otherpacs" ] ],
#              ]

'''


CONF_PATH = os.environ.get('DCM_CONF_PATH', 
                           os.path.join(click.get_app_dir('dcm'), 
                                        'dcm_conf.toml'))



class DcmConfig:
    '''Capture config and support mixing with external (eg. CLI) options

    The variouse `get_*` methods will tranparently handle inputs that are 
    named references from the config file, inline configuration strings, or 
    configuration dicts. 
    '''
    def __init__(self, 
                 config_path: os.PathLike = CONF_PATH, 
                 create_if_missing: bool = False):
        self._config_path = config_path
        if not os.path.exists(config_path):
            if create_if_missing:
                config_dir = os.path.dirname(config_path)
                if not os.path.exists(config_dir):
                    os.makedirs(config_dir)
                with open(config_path, 'w') as f:
                    f.write(_default_conf)
                conf_str = _default_conf
            else:
                raise FileNotFoundError(config_path)
        else:
            with open(config_path, 'r') as f:
                conf_str = f.read()
        
        # Read the raw TOML contents
        self._raw_conf = toml.loads(conf_str)

        # Parse what we can ahead of time info this dict
        self._conf = {}

        # Pull out sections from raw config data and do sanity checks
        raw_local = self._raw_conf.get('local_nodes', {})
        raw_remote = self._raw_conf.get('remote_nodes', {})
        raw_dirs = self._raw_conf.get('local_dirs', {})
        raw_remote_repos = self._raw_conf.get('remote_repos', {})
        raw_selectors = self._raw_conf.get('selectors', {})
        raw_static_routes = self._raw_conf.get('static_routes', {})
        raw_dyn_routes = self._raw_conf.get('dynamic_routes', {})
        conflicts = set()
        for section in (raw_remote, raw_dirs, raw_remote_repos, raw_static_routes, raw_dyn_routes):
            conflicts &= set(raw_local) | set(section)
        if conflicts:
            raise InvalidConfigError(f"Name conflicts: {conflicts}")
        conflicts = set()
        for section in (raw_remote, raw_remote_repos, raw_static_routes, raw_dyn_routes):
            conflicts &= set(raw_dirs) | set(section)
        if conflicts:
            raise InvalidConfigError(f"Name conflicts: {conflicts}")
        conflicts = set(raw_dyn_routes) | set(raw_static_routes) 
        if conflicts:
            raise InvalidConfigError(f"Name conflicts: {conflicts}")

        # We can pre-convert local / remote DcmNodes and Selectors

        # Convert local_nodes section and determine default local node
        self._local_nodes = {}
        self._default_local = None
        def_local_name = None
        for local_name, local_val in raw_local.items():
            if local_name == 'default' or def_local_name is None:
                def_local_name = local_name
            if isinstance(local_val, dict) and 'host' not in local_val:
                local_val['host'] = '0.0.0.0'
            try:
                self._local_nodes[local_name] = DcmNode.from_toml_val(local_val)
            except Exception as e:
                raise InvalidConfigError(f"Error parsing local_node '{local_name}': {e}'")
        if def_local_name is not None:
            self._default_local = self._local_nodes[def_local_name]
        
        # Convert remote_nodes section
        try:
            self._remote_nodes = {k: DcmNode.from_toml_val(v) 
                                  for k, v in raw_remote.items()}
        except Exception as e:
            raise InvalidConfigError(f"Error parsing 'remote_nodes' section: {e}")

        # Convert selectors section
        self._selectors = {}
        for sel_name, sel_val in raw_selectors.items():
            if not hasattr(sel_val, 'all_of'):
                sel = SingleSelector.from_toml_val(sel_val)
            else:
                for attr in ('none_of', 'all_of', 'any_of'):
                    sel_list = [self.get_selector(name) 
                                for name in getattr(sel_val, attr)]
                    setattr(sel_val, attr, sel_list)
                sel = MultiSelector.from_toml_val(sel_val)
            self._selectors[sel_name] = sel

        # We will build these dynamically to incoporate CLI options
        self._net_repos = deepcopy(raw_remote_repos)
        self._local_dirs = deepcopy(raw_local_dirs)
        self._static_routes = deepcopy(raw_static_routes)
        self._dynamic_routes = deepcopy(raw_dyn_routes)

        self._net_repo_kwargs = {}
        self._local_dir_kwargs = {}
        self._static_route_kwargs = {}
        self._dynamic_route_kwargs = {}

    def set_net_repo_kwargs(**kwargs) -> None:
        '''Override parameters for any NetRepo built'''
        self._net_repo_kwargs = kwargs
    
    def set_local_dir_kwargs(**kwargs) -> None:
        '''Override parameters for any LocalDir built'''
        self._local_dir_kwargs = kwargs

    def set_static_route_kwargs(**kwargs) -> None:
        '''Override parameters for any StaticRoute'''
        self._static_route_kwargs = kwargs
    
    def set_dynamic_route_kwargs(**kwargs) -> None:
        '''Override any parameters for SelectorDestMap'''
        self._dynamic_route_kwargs = kwargs

    @property
    def default_local(self) -> DcmNode:
        '''The default local DcmNode to use'''
        return self._default_local

    def _get_node(in_val: Union[str, Dict[str, Any]], 
                  nodes: Dict[str, DcmNode] = None) -> DcmNode:
        '''Return named node or convert value to DcmNode'''
        res = nodes.get(in_val)
        if res is not None:
            return res
        return DcmNode.from_toml_val(in_val)

    def get_local_node(self, 
                       in_val: Optional[Union[str, Dict[str, Any]]]
                       ) -> DcmNode:
        '''Get local DcmNode corresponding to `in_val`'''
        if in_val is None:
            return self._default_local
        return self._get_node(in_val, self._local_nodes)

    def get_remote_node(self,
                        in_val: Union[str, Dict[str, Any]]
                        ) -> DcmNode:
        '''Get remote DcmNode corresponding to `in_val`'''
        if isinstnace(in_val, str) and in_val in self._net_repos:
            repo = self.get_net_repo(in_val)
            return repo.remote
        return self._get_node(in_val, self._remote_nodes)
    
    def get_selector(self,
                     in_val: Union[str, Dict[str, Any]]
                     ) -> Selector:
        '''Get a Selector corresponding to `in_val`'''
        if isinstnace(in_val, str):
            res = self._selectors.get(in_val)
            if res is not None:
                return res
        return SingleSelector.from_toml_val(in_val)
    
    def get_net_repo(self,
                     in_val: Union[str, Dict[str, Any]]) -> NetRepo:
        '''Get a NetRepo corresponding to `in_val`'''
        if isinstance(in_val, str):
            raw_dict = self._net_repos.get(in_val)
            if raw_dict is None:
                remote = self.get_remote_node(in_val)
                raw_dict = {'remote': remote}
        else:
            raw_dict = deepcopy(in_val)
        raw_dict.update(self._net_repo_kwargs)
        return NetRepo.from_toml_dict(raw_dict)

    def get_local_dir(self,
                      in_val: Union[str, Dict[str, Any]]) -> LocalDir:
        '''Get a LocalDir corresponding to `in_val`'''
        if isinstance(in_val, str):
            raw_dict = self._local_dirs.get(in_val)
            if raw_dict is None:
                raw_dict = LocalDir.inline_to_dict(in_val)
        else:
            raw_dict = deepcopy(in_val)
        raw_dict.update(self._local_dir_kwargs)
        return LocalDir.from_toml_dict(raw_dict)

    def get_bucket(self,
                 in_val: Union[str, Dict[str, Any]]) -> Union[NetRepo, LocalDir]:
        # If a named net_repo exists we prefer that over a path of the same name
        if instance(in_val, str):
            if in_val in self._net_repos or in_val in self._remote_nodes:
                return self.get_net_repo(in_val)
        try:
            return get_local_dir(in_val)
        except Exception:
            # We fall back to an inline NetRepo if the path doesn't exist
            return get_net_repo(in_val)

    def _merge_route_kwargs(self, kwargs, filt):
        if filt is None or kwargs.get('filt') is None:
            return kwargs
        kwargs = deepcopy(kwargs)
        kwargs['filt'] = MultiFilter((kwargs['filt'], filt))
        return kwargs

    def get_route(self,
                  in_val: Union[str, Dict[str, Any]]) -> Union[StaticRoute, DynamicRoute]:
        if instance(in_val, str):
            if in_val in self._dynamic_routes:
                kwargs = deepcopy(self._dynamic_routes[in_val])
                kwargs.update(self._merge_route_kwargs(self._dynamic_route_kwargs, 
                                                       kwargs.get('filt')))
                rmap = []
                for sel, dests in dr_dict['routing_map']:
                    rmap.append((self.get_selector(sel), 
                                 tuple(self.get_bucket(d, conf_dict, default_local) 
                                       for d in dests
                                       )
                                ))
                filt = kwargs.get('filt')
                if filt is not None:
                    kwargs['filt'] = self.get_selector(filt).get_filter()
                return SelectorDestMap(**kwargs).get_dynamic_route()
            else:
                if in_val in self._static_routes:
                    kwargs = deepcopy(self._static_routes[in_val])
                else:
                    kwargs = {'dests': (in_val,)}
                kwargs.update(self._merge_route_kwargs(self._static_route_kwargs, 
                                                       kwargs.get('filt')))
                kwargs['dests'] = tuple(self.get_bucket(d) for d in kwargs['dests'])
                filt = kwargs.get('filt')
                if filt is not None:
                    kwargs['filt'] = self.get_selector(filt).get_filter()
                return StaticRoute(**kwargs)
        else:
            kwargs = deepcopy(in_val)
            if 'routing_map' in kwargs:
                kwargs.update(self._merge_route_kwargs(self._dynamic_route_kwargs, 
                                                       kwargs.get('filt')))
                return SelectorDestMap.from_toml_dict(kwargs).get_dynamic_route()
            else:
                kwargs.update(self._merge_route_kwargs(self._static_route_kwargs, 
                                                       kwargs.get('filt')))
                return StaticRoute.from_toml_dict(kwargs)
    
    def get_routes(self,
                   in_vals: List[Union[str, Dict[str, Any]]]
                   ) -> List[Union[StaticRoute, DynamicRoute]]:
        res = []
        plain_dests = []
        for in_val in in_vals:
            if in_val in self._static_routes or in_val in self._dynamic_routes:
                res.append(self.get_route(in_val))
            else:
                plain_dests.append(self.get_bucket(in_val))
        if plain_dests:
            res.append(StaticRoute(tuple(plain_dests)), **self._static_route_kwargs)
        return res

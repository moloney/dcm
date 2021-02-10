# Config parsing
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, MutableMapping, Set

import toml
import click

from .util import TomlConfigurable, InlineConfigurable, PathInputType
from .net import DcmNode, LocalEntity
from .store.local_dir import LocalDir
from .store.net_repo import NetRepo
from .filt import Selector, SingleSelector, MultiSelector, Filter, MultiFilter
from .route import StaticRoute, DynamicRoute, SelectorDestMap


class InvalidConfigError(Exception):
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
#  [static_routes.movescu_pacs]
#  dests = [ "yourpacs", "otherpacs" ]
#  methods = [ "REMOTE_COPY" ]
#
#  # Define simple selector inline and use it as a filter
#  [static_routes.filt_mypacs]
#  dests = [ "yourpacs" ]
#  filt = "StudyDescription == ProjectA"
#
#[dynamic_routes]
#
#  # We can specify a dynamic route by groups selectors with dests
#  [dynamic_routes.dyn_route]
#  routing_map = { internal = [ "yourpacs" ], external =  [ "otherpacs" ] }
#
'''


CONF_PATH = os.environ.get('DCM_CONF_PATH', 
                           os.path.join(click.get_app_dir('dcm'), 
                                        'dcm_conf.toml'))


CONFIG_VERSION = 2


def get_version(data: MutableMapping[str, Any]) -> int:
    if 'local_node' in data:
        return 1
    return 2


def migrate(version: int, data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    if version < 2:
        local = data.get('local_node')
        if local:
            del data['local_node']
            data['local_nodes'] = {'default': local}
    return data


class DcmConfig:
    '''Capture config and support mixing with external (eg. CLI) options

    The variouse `get_*` methods will tranparently handle inputs that are 
    named references from the config file, inline configuration strings, or 
    configuration dicts. 
    '''
    def __init__(self, 
                 config_path: PathInputType = CONF_PATH, 
                 create_if_missing: bool = False):
        self._config_path = Path(config_path)
        if not self._config_path.exists():
            if create_if_missing:
                config_dir = self._config_path.parent
                config_dir.mkdir(parents=True, exist_ok=True)
                with self._config_path.open('w') as f:
                    f.write(_default_conf)
                conf_str = _default_conf
            else:
                raise FileNotFoundError(self._config_path)
        else:
            with self._config_path.open('r') as f:
                conf_str = f.read()
        
        # Read the raw TOML contents
        self._raw_conf = toml.loads(conf_str)

        # Handle migrations
        version = get_version(self._raw_conf)
        if version != CONFIG_VERSION:
            self._raw_conf = migrate(version, self._raw_conf)
            with self._config_path.open('w') as f:
                toml.dump(self._raw_conf, f)

        # Parse what we can ahead of time info this dict
        self._conf: MutableMapping[str, Any] = {}

        # Pull out sections from raw config data and do sanity checks
        raw_local = self._raw_conf.get('local_nodes', {})
        raw_remote = self._raw_conf.get('remote_nodes', {})
        raw_dirs = self._raw_conf.get('local_dirs', {})
        raw_remote_repos = self._raw_conf.get('remote_repos', {})
        raw_selectors = self._raw_conf.get('selectors', {})
        raw_static_routes = self._raw_conf.get('static_routes', {})
        raw_dyn_routes = self._raw_conf.get('dynamic_routes', {})
        conflicts: Set[str] = set()
        for section in (raw_remote, raw_dirs, raw_remote_repos, raw_static_routes, raw_dyn_routes):
            conflicts |= set(raw_local) & set(section)
        if conflicts:
            raise InvalidConfigError(f"Name conflicts: {conflicts}")
        conflicts = set()
        for section in (raw_remote, raw_remote_repos, raw_static_routes, raw_dyn_routes):
            conflicts |= set(raw_dirs) & set(section)
        if conflicts:
            raise InvalidConfigError(f"Name conflicts: {conflicts}")
        conflicts = set(raw_dyn_routes) & set(raw_static_routes) 
        if conflicts:
            raise InvalidConfigError(f"Name conflicts: {conflicts}")

        # We can pre-convert local / remote DcmNodes and Selectors

        # Convert local_nodes section and determine default local node
        self._local_nodes: Dict[str, DcmNode] = {}
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
        self._selectors: MutableMapping[str, Union[SingleSelector, MultiSelector]] = {}
        for sel_name, sel_val in raw_selectors.items():
            if isinstance(sel_val, str) or 'attr' in sel_val:
                self._selectors[sel_name] = SingleSelector.from_toml_val(sel_val)
            else:
                for key in ('none_of', 'all_of', 'any_of'):
                    if key in sel_val:
                        sel_list = [self.get_selector(name) 
                                    for name in sel_val[key]]
                        sel_val[key] = sel_list
                self._selectors[sel_name] = MultiSelector.from_toml_val(sel_val)

        # We will build these dynamically to incoporate CLI options
        self._net_repos = deepcopy(raw_remote_repos)
        self._local_dirs = deepcopy(raw_dirs)
        self._static_routes = deepcopy(raw_static_routes)
        self._dynamic_routes = deepcopy(raw_dyn_routes)

        self._net_repo_kwargs: Dict[str, Any] = {}
        self._local_dir_kwargs: Dict[str, Any] = {}
        self._static_route_kwargs: Dict[str, Any] = {}
        self._dynamic_route_kwargs: Dict[str, Any] = {}

    def set_net_repo_kwargs(self, **kwargs: Dict[str, Any]) -> None:
        '''Override parameters for any NetRepo built'''
        self._net_repo_kwargs = kwargs
    
    def set_local_dir_kwargs(self, **kwargs: Dict[str, Any]) -> None:
        '''Override parameters for any LocalDir built'''
        self._local_dir_kwargs = kwargs

    def set_static_route_kwargs(self, **kwargs: Dict[str, Any]) -> None:
        '''Override parameters for any StaticRoute'''
        self._static_route_kwargs = kwargs
    
    def set_dynamic_route_kwargs(self, **kwargs: Dict[str, Any]) -> None:
        '''Override any parameters for SelectorDestMap'''
        self._dynamic_route_kwargs = kwargs

    @property
    def default_local(self) -> Optional[DcmNode]:
        '''The default local DcmNode to use'''
        return self._default_local

    def _get_node(self,
                  in_val: Union[str, Dict[str, Any]], 
                  nodes: Dict[str, DcmNode]) -> DcmNode:
        '''Return named node or convert value to DcmNode'''
        if isinstance(in_val, str):
            res = nodes.get(in_val)
            if res is not None:
                return res
        return DcmNode.from_toml_val(in_val)

    def get_local_node(self, 
                       in_val: Optional[Union[str, Dict[str, Any]]]
                       ) -> DcmNode:
        '''Get local DcmNode corresponding to `in_val`'''
        if in_val is None:
            if self._default_local is None:
                raise ValueError("No local nodes defined")
            return self._default_local
        return self._get_node(in_val, self._local_nodes)

    def get_remote_node(self,
                        in_val: Union[str, Dict[str, Any]]
                        ) -> DcmNode:
        '''Get remote DcmNode corresponding to `in_val`'''
        if isinstance(in_val, str) and in_val in self._net_repos:
            repo = self.get_net_repo(in_val)
            return repo.remote
        return self._get_node(in_val, self._remote_nodes)
    
    def get_selector(self,
                     in_val: Union[str, Dict[str, Any]]
                     ) -> Selector:
        '''Get a Selector corresponding to `in_val`'''
        if isinstance(in_val, str):
            res = self._selectors.get(in_val)
            if res is not None:
                return res
        return SingleSelector.from_toml_val(in_val)
    
    def get_net_repo(self,
                     in_val: Union[str, Dict[str, Any]]) -> NetRepo:
        '''Get a NetRepo corresponding to `in_val`'''
        if isinstance(in_val, str):
            raw_dict = deepcopy(self._net_repos.get(in_val))
            if raw_dict is None:
                raw_dict = {'remote': in_val}
        else:
            raw_dict = deepcopy(in_val)
        if 'local' not in raw_dict:
            raw_dict['local'] = self.default_local
        else:
            raw_dict['local'] = self.get_local_node(raw_dict['local'])
        raw_dict['remote'] = self.get_remote_node(raw_dict['remote'])
        raw_dict.update(self._net_repo_kwargs)
        return NetRepo.from_toml_dict(raw_dict)

    def get_local_dir(self,
                      in_val: Union[str, Dict[str, Any]]) -> LocalDir:
        '''Get a LocalDir corresponding to `in_val`'''
        if isinstance(in_val, str):
            raw_dict = deepcopy(self._local_dirs.get(in_val))
            if raw_dict is None:
                raw_dict = LocalDir.inline_to_dict(in_val)
        else:
            raw_dict = deepcopy(in_val)
        raw_dict.update(self._local_dir_kwargs)
        return LocalDir.from_toml_dict(raw_dict)

    def get_bucket(self,
                 in_val: Union[str, Dict[str, Any]]) -> Union[NetRepo, LocalDir]:
        if isinstance(in_val, str):
            # Prefer named net nodes / repos
            if in_val in self._net_repos or in_val in self._remote_nodes:
                return self.get_net_repo(in_val)
            # Check for named local_dir or if it looks path like
            elif (in_val in self._local_dirs or 
                  os.sep in in_val or 
                  '{' in in_val or 
                  os.path.exists(in_val.split(':')[0])):
                return self.get_local_dir(in_val)
            # We fall back to an inline NetRepo
            return self.get_net_repo(in_val)
        else:
            if 'path' in in_val:
                return self.get_local_dir(in_val)
            return self.get_net_repo(in_val)

    def _merge_route_kwargs(self, 
                            kwargs: Dict[str, Any], 
                            filt: Filter) -> Dict[str, Any]:
        if filt is None or kwargs.get('filt') is None:
            return kwargs
        kwargs = deepcopy(kwargs)
        kw_filt = kwargs['filt']
        assert isinstance(kw_filt, Filter)
        kwargs['filt'] = MultiFilter(filters=(kw_filt, filt))
        return kwargs

    def get_static_route(self,
                         in_val: Union[str, Dict[str, Any]]
                         ) -> StaticRoute:
        if isinstance(in_val, str):
            if in_val in self._static_routes:
                kwargs = deepcopy(self._static_routes[in_val])
            else:
                raise ValueError("Invalid input")
        else:
            kwargs = deepcopy(in_val)
        filt = kwargs.get('filt')
        if filt is not None:
            kwargs['filt'] = self.get_selector(filt).get_filter()
        kwargs.update(self._merge_route_kwargs(self._static_route_kwargs, 
                                               kwargs.get('filt')))
        kwargs['dests'] = tuple(self.get_bucket(d) for d in kwargs['dests'])
        return StaticRoute.from_toml_dict(kwargs)

    def get_selector_dest_map(self,
                              in_val: Union[str, Dict[str, Any]]
                              ) -> SelectorDestMap:
        if isinstance(in_val, str):
            if in_val in self._dynamic_routes:
                kwargs = deepcopy(self._dynamic_routes[in_val])
            else:
                raise ValueError("Invalid input")
        else:
            kwargs = deepcopy(in_val)
        filt = kwargs.get('filt')
        if filt is not None:
            kwargs['filt'] = self.get_selector(filt).get_filter()
        kwargs.update(self._merge_route_kwargs(self._dynamic_route_kwargs, 
                                               kwargs.get('filt')))
        rmap = []
        for sel, dests in kwargs['routing_map'].items():
            rmap.append((self.get_selector(sel), 
                         tuple(self.get_bucket(d) for d in dests)
                         ))
        kwargs['routing_map'] = tuple(rmap)
        return SelectorDestMap.from_toml_dict(kwargs)

    def get_route(self,
                  in_val: Union[str, Dict[str, Any]]) -> Union[StaticRoute, DynamicRoute]:
        try: 
            return self.get_static_route(in_val)
        except:
            return self.get_selector_dest_map(in_val).get_dynamic_route()
    
    def get_routes(self,
                   in_vals: List[Union[str, Dict[str, Any]]]
                   ) -> List[Union[StaticRoute, DynamicRoute]]:
        res: List[Union[StaticRoute, DynamicRoute]] = []
        plain_dests = []
        for in_val in in_vals:
            if in_val in self._static_routes:
                res.append(self.get_static_route(in_val))
            elif in_val in self._dynamic_routes:
                res.append(self.get_selector_dest_map(in_val).get_dynamic_route())
            else:
                plain_dests.append(self.get_bucket(in_val))
        if plain_dests:
            res.append(StaticRoute(tuple(plain_dests), **self._static_route_kwargs))
        return res

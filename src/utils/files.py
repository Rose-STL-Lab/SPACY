import os
from datetime import datetime
from src.data.SyntheticDataConfig import SyntheticDataConfig
from src.data.SSTDataConfig import SSTDataConfig

def generate_run_name(cfg):
    """
    Generate a unique run name for a given config file
    :param cfg: OmegaConf
        The configuration file to use
    """
    # TODO
    model_name = cfg.model.model._target_.split(".")[-1]
    data_name = generate_datafolder_name(cfg.data.data_config)
    seed = cfg.random_seed
    
    run_name = f"model={model_name}_data={data_name}_seed={seed}"

    # run_name = f"model={model_name}_seed={seed}_data={data_name}"
    # run_name += datetime.now().strftime("%Y%m%d_%H_%M_%S")
    return run_name

def generate_sst_datafolder_name(cfg: SSTDataConfig):
    # if cfg.hist_dep_noise:
    #     noise_type = cfg.hist_dep_noise_type
    # else:
    #     noise_type = cfg.base_noise_type
    
    # datafolder_name = f"{cfg.data_type}_" \
    #                   f"num_nodes={cfg.num_nodes}_nx={cfg.nx}_ny={cfg.ny}_" \
    #                   f"num_variates={cfg.num_variates}_" \
    #                   f"functional_relationships={cfg.functional_relationships}_lag={cfg.lag}_" \
    #                   f"time_length={cfg.time_length}"

    # if cfg.model == 'infer_spacy':
    #     datafolder_name = f"{datafolder_name}_{cfg.seed}"

    sst_datafolder = {'sst_rmseason': 'control_sst_rmseason', 'temp_rmseason': 'control_temp_rmseason', 'preci_rmseason':'control_precip_rmseason', 'sst_inst': 'sst_inst'}
    # if cfg.num_variates > 1 and cfg.model == 'sst':
    #     datafolder_name = (sst_datafolder['sst_rmseason'],sst_datafolder['preci_rmseason'])
    # else:
    datafolder_name = sst_datafolder[cfg.model]

    return datafolder_name

def generate_synthetic_datafolder_name(cfg: SyntheticDataConfig):
    if cfg.hist_dep_noise:
        noise_type = cfg.hist_dep_noise_type
    else:
        noise_type = cfg.base_noise_type

    # datafolder_name = f"{cfg.model}_inst_graph_type={cfg.inst_graph_type}_lag_graph_type={cfg.lag_graph_type}_" \
    #                   f"num_nodes={cfg.num_nodes}_nx={cfg.nx}_ny={cfg.ny}_node_dist={cfg.node_dist}_" \
    #                   f"num_variates={cfg.num_variates}_" \
    #                   f"functional_relationships={cfg.functional_relationships}_lag={cfg.lag}_" \
    #                   f"noise_type={noise_type}_disable_inst={cfg.disable_inst}_hist_dep_noise={cfg.hist_dep_noise}_" \
    #                   f"time_length={cfg.time_length}_map_type={cfg.map_type}"

    datafolder_name = f"{cfg.model}_inst_graph_type={cfg.inst_graph_type}_lag_graph_type={cfg.lag_graph_type}_" \
                      f"num_nodes={cfg.num_nodes}_nx={cfg.nx}_ny={cfg.ny}_node_dist={cfg.node_dist}_" \
                      f"num_variates={cfg.num_variates}_" \
                      f"functional_relationships={cfg.functional_relationships}_lag={cfg.lag}_" \
                      f"noise_type={noise_type}_hist_dep_noise={cfg.hist_dep_noise}_" \
                      f"time_length={cfg.time_length}_map_type={cfg.map_type}"
    
    if cfg.disable_inst:
        datafolder_name = f'{datafolder_name}_disable_inst'

    if cfg.num_variates > 1:
        if cfg.disjoint_nodes:
            datafolder_name = f'{datafolder_name}_dv'

    if cfg.model == 'infer_spacy':
        datafolder_name = f"{datafolder_name}_{cfg.seed}"


    return datafolder_name

def generate_datafolder_name(cfg):
    if 'sst' in cfg.model:
        return generate_sst_datafolder_name(cfg)
    else:
        return generate_synthetic_datafolder_name(cfg)

def make_dirs(directory):
    """
    Create directories if they do not exist.

    :param directory: str
        The directory path to be created.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

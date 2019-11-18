#!/usr/bin/env python2.7

import argparse
import bnpy
import logging
import numpy as np
import os
import pandas as pd
import shutil
import errno

def ens_to_hugo(ens_to_hugo_filepath='/opt/typhon/data/EnsGeneID_Hugo_Observed_Conversions.txt'):
    ens_to_hugo_dict = {}
    with open(ens_to_hugo_filepath) as f:
        header = next(f)
        for line in f:
            hugo, ens = line.strip().split('\t')
            ens_to_hugo_dict[ens] = hugo
    return ens_to_hugo_dict

# mkdir_p from hydra library/utils.py
def mkdir_p(path):
    """
    https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    :param path:
    :return:
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# get_assignments from hydra library/fit.py
def get_assignments(model, data):
    """
    Takes model and data and classifies samples
    Will label samples with NaN if they do not
    fit in any of the model components
    :param model:
    :param data:
    :return:
    """
    unclass = 1 - np.sum(model.allocModel.get_active_comp_probs())
    # Get the sample assignments
    LP = model.calc_local_params(data)
    asnmts = []
    for row in range(LP['resp'].shape[0]):
        _max = np.max(LP['resp'][row, :])
        if _max < unclass:
            asnmts.append(np.nan)

        else:
            _arg = np.argmax(LP['resp'][row, :])
            asnmts.append(_arg)

    return asnmts

def fit_models(data, diagnosis, output_dir, models_root_dir='/opt/typhon/models/'):
    logger = logging.getLogger('root')
    models_pth = os.path.join(models_root_dir, diagnosis)

    # Load Enrichment Analysis
    for model in os.listdir(models_pth):
        logger.info("Applying %s model" % model)
        model_pth = os.path.join(models_pth, model, model)
        hmodel = bnpy.ioutil.ModelReader.load_model_at_prefix(model_pth,
                                                              prefix=model)
        train_pth = os.path.join(models_pth, model, model, 'training-data.tsv')
        train_data = pd.read_csv(train_pth, sep='\t', index_col=0)
        _data = data.reindex(train_data.index).values
        xdata = bnpy.data.XData(_data.reshape(len(_data), 1))
        assignment = get_assignments(hmodel, xdata).pop()
        logger.debug("Place in cluster %d" % assignment)
        if pd.isnull(assignment):
            logger.info("WARNING: Could not classify sample!")
            continue

        output_dir = os.path.join(output_dir, model)
        mkdir_p(output_dir)
        feature_src = os.path.join(os.path.join(models_pth, model, 'features', str(assignment)))
        feature_dest = os.path.join(output_dir, 'CLUSTER%d' % assignment)
        shutil.copytree(feature_src, feature_dest)


def main():
    """
    Typhon Pipeline
    """
    parser = argparse.ArgumentParser(description=main.__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--diagnosis',
                        help='Patient diagnosis',
                        required=True)

    parser.add_argument('--RSEM',
                        help='Path to N-of-1 RSEM file.',
                        required=True)

    parser.add_argument('-o', '--output-dir',
                        help='Output directory',
                        default='typhon-output')

    parser.add_argument('--debug',
                        action='store_true',
                        default=False)

    args = parser.parse_args()

    available_models = ['MYCN-NA-Neuroblastoma']

    if args.diagnosis not in available_models:
        msg = "Please select one of the following diagnoses:\n%s" % '\n'.join(available_models)
        raise ValueError(msg)

    # Set up logger
    level = logging.INFO

    # Make the output directory if it doesn't already exist
    mkdir_p(args.output_dir)

    logging.basicConfig(filename=os.path.join(args.output_dir, 'typhon.log'),
                        level=level)
    logging.getLogger().addHandler(logging.StreamHandler())

    logger = logging.getLogger('root')

    data = pd.read_csv(args.RSEM, sep='\t')

    data['hugo'] = data['gene_id'].map(ens_to_hugo())
    tpm = data.reindex(['hugo', 'TPM'], axis=1).groupby('hugo').sum()
    exp = np.log2(tpm + 1)

    logger.info("Starting run...")
    fit_models(exp, args.diagnosis, args.output_dir)

if __name__ == '__main__':
    main()


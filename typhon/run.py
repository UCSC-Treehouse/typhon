#!/usr/bin/env python2.7

import sys
sys.path.append('/opt/hydra/')

import argparse
import bnpy
import logging
import numpy as np
import os
import pandas as pd
import shutil

import library.analysis as hydra
from library.utils import mkdir_p
from library.fit import get_assignments


ens_to_hugo = {}
with open('/opt/typhon/data/EnsGeneID_Hugo_Observed_Conversions.txt') as f:
    header = next(f)
    for line in f:
        hugo, ens = line.strip().split('\t')
        ens_to_hugo[ens] = hugo


def fit_models(data, diagnosis, output_dir):
    logger = logging.getLogger('root')
    models_pth = os.path.join('/opt/typhon/models/', diagnosis)

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

    data['hugo'] = data['gene_id'].map(ens_to_hugo)
    tpm = data.reindex(['hugo', 'TPM'], axis=1).groupby('hugo').sum()
    exp = np.log2(tpm + 1)

    logger.info("Starting run...")
    fit_models(exp, args.diagnosis, args.output_dir)

if __name__ == '__main__':
    main()


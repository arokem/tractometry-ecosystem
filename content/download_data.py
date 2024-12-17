
# Set environment to point to local data directory for dowonloading templates:

import os
import os.path as op
pwd = os.getcwd()
os.environ["TEMPLATEFLOW_HOME"] = op.join(pwd, "..", "data_", "tractometry")
os.environ["DIPY_HOME"] = op.join(pwd, "..", "data_", "tractometry")
os.environ["AFQ_HOME"] = op.join(pwd, "..", "data_", "tractometry")


from AFQ.data.fetch import (
        read_templates,
        read_pediatric_templates,
        read_callosum_templates,
        read_cp_templates,
        read_or_templates,
        read_ar_templates)

import templateflow.api as tflow


def download_templates():
    read_templates()
    read_pediatric_templates()
    read_callosum_templates()
    read_cp_templates()
    read_or_templates()
    read_ar_templates()


# Templates:
tflow.get('MNI152NLin2009cAsym',
          resolution=1,
          desc='brain',
          suffix='mask')

download_templates()
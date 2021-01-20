# -*- coding: utf-8 -*-

"""

@author: bartulem

Compare tuning-curve rate differences in weight/no-weight sessions.

"""

import numpy as np
import os
import scipy.io



class WeightComparer:

    def __init__(self, light_one):
        self.light_one = light_one

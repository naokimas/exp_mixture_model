# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
import sys
import os
from exp_mixture_model import EMM, EMMs


def _read_data_from_stdin():
    """
    Read data from stdin.
    """
    lines = []
    while (True):
        line = sys.stdin.readline()
        if line == "":
            break
        lines.append(line)
    return np.array(lines, dtype=float)


def _plot_result(fname, plot_func):
    """
    Plot result with 'plot_func' and save it as 'fname'.

    Parameters
    -------
    fname : str
        Name of the figure file to be saved.
    plot_func : function
        Function for plotting the results.
    """
    fig, ax = plt.subplots()
    plot_func(ax, show_flag=False)

    count = 2
    fn_only, ext = os.path.splitext(fname)
    while os.path.exists(fname):
        fname = "%s(%d)%s" % (fn_only, count, ext)
        count += 1

    fig.savefig(fname, format="png", bbox_inches='tight')
    print("Output %s" % fname)


parser = OptionParser(usage="usage: %prog  [--file INPUT_FILENAME] [options]")
parser.add_option("-v", "--verbose",
                  action="store_true", dest="verbose", default=True,
                  help="Print all info")
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose",
                  help="Be quiet")
parser.add_option("-d", "--data",
                  action="store_true", dest="data", default=False,
                  help="Read input data from stdin")
parser.add_option("-f", "--file",
                  default=None, dest="input_file",
                  help="Read input data from file")
parser.add_option("-k", "--k_initial",
                  default=None, dest="k", type="int",
                  help="Set a single k value (i.e. the number of components). Then, fit just one EMM.")
parser.add_option("-c", "--criterion",
                  default=None, dest="criterion",
                  help="Specify the model selection criterion. Choose from 'marginal_log_likelihood',"
                       " 'joint_log_likelihood', 'AIC', 'BIC', 'AIC_LVC', 'BIC_LVC', 'NML_LVC', or 'DNML'")
parser.add_option("-o", "--output",
                  default="emmfit", dest="output",
                  help="Set prefix of the output filename")


if __name__ == "__main__":
    (options, args) = parser.parse_args()

    if options.input_file is not None:
        x = np.loadtxt(options.input_file)
    elif options.data:
        x = _read_data_from_stdin()
    else:
        raise Exception("Input data to be fitted. Provide either --file or --data")

    if options.k is None:
        if options.criterion not in [None, "marginal_log_likelihood", "joint_log_likelihood",
                                     "AIC", "BIC", "AIC_LVC", "BIC_LVC", "NML_LVC", "DNML"]:
            raise ValueError(
                """Choose 'criterion' from ['marginal_log_likelihood', 'joint_log_likelihood',
                'AIC', 'BIC', 'AIC_LVC', 'BIC_LVC', 'NML_LVC', 'DNML'].
                """)

        emms = EMMs()
        emms.fit(x, verbose=options.verbose)

        if options.criterion is not None:
            best_model = emms.select(options.criterion)

            print("%s selects" % options.criterion)
            best_model.print_result()

            _plot_result("%s_%s_survival_probability.png" % (options.output, options.criterion),
                         best_model.plot_survival_probability)
            _plot_result("%s_%s_odds_ratio.png" % (options.output, options.criterion),
                         best_model.plot_odds_ratio)
        else:
            for criterion in ["AIC", "BIC", "AIC_LVC", "BIC_LVC", "NML_LVC", "DNML"]:
                best_model = emms.select(criterion)

                print("%s selects" % criterion)
                best_model.print_result()

        if options.verbose:
            emms.print_result_table()
    else:
        emm = EMM(options.k)
        emm.fit(x)

        # print results
        emm.print_result()
        _plot_result("%s_survival_probability.png" % options.output,
                     emm.plot_survival_probability)
        _plot_result("%s_odds_ratio.png" % options.output,
                     emm.plot_odds_ratio)

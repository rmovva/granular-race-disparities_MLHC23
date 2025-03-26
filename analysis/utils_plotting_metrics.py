import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
# plt.rcParams["font.family"] = "Arial CE"
from matplotlib.patches import Patch
import seaborn as sns

from race_categories import coarse_races, granular_to_coarse, granular_abbrev, coarse_to_granular, coarse_abbrev
import warnings
warnings.filterwarnings("ignore", message="FixedFormatter should only be used together with FixedLocator")

outcomes_to_labels = {
    'outcome_hospitalization': 'Outcome: Hospitalization',
    'outcome_critical': 'Outcome: ICU or Mortality',
    'outcome_ed_revisit_3d': 'Outcome: ED Revisit',
}

outcome_labels_short = {
    'outcome_hospitalization': 'Hospitalization',
    'outcome_critical': 'Critical',
    'outcome_ed_revisit_3d': 'Revisit',
}

metrics_to_labels = {
    'auprc': 'AUPRC',
    'auroc': 'AUROC',
    'fpr_fixed': 'False Positive Rate',
    'fnr_fixed': 'False Negative Rate',
    'fpr': 'False Positive Rate',
    'fnr': 'False Negative Rate',
    'mean_pred_pos': r'Mean prediction, true $y$ = 1',
    'mean_pred_neg': r'Mean prediction, true $y$ = 0',
    '1_bins': "Exp. Calibration Error (1-bin)",
    '5_bins': "Exp. Calibration Error (5-bins)",
    '10_bins': "Exp. Calibration Error (10-bins)",
    'brier': "Brier Calibration",
}

risk_scores_to_labels = {
    'triage_acuity': 'Triage Severity',
    'score_NEWS': 'National Early Warning Score (NEWS)',
    'score_CART': 'Cardiac Arrest Risk Triage (CART)',
}

coarse_palettes = {
        'ASIAN': 'Blues',
        'BLACK/AFRICAN AMERICAN': 'Greens',
        'HISPANIC OR LATINO': 'Reds',
        'OTHER': 'YlOrBr',
        'WHITE': 'Purples',
}

# Pastel hues of red, green, blue, purple
colors = ['#f44336', '#388e3c', '#2196f3', '#9c27b0']
coarse_colors = {
    'ASIAN': colors[1],
    'BLACK/AFRICAN AMERICAN': colors[2],
    'HISPANIC OR LATINO': colors[0],
    'WHITE': colors[3],
}

outcomes_all = ['outcome_hospitalization', 'outcome_critical', 'outcome_ed_revisit_3d']


def evaluate_calibration(yhat, y, n_bins=10):
    """
    Evaluate calibration of a set of predictions yhat, given true labels y.
    Uses expected calibration error (ECE) across N bins, default N=10.
    See e.g. https://www.dbmi.pitt.edu/wp-content/uploads/2022/10/Obtaining-well-calibrated-probabilities-using-Bayesian-binning.pdf.
    """
    # Set bounds to be equally spaced percentiles of yhat
    bounds = np.percentile(yhat, np.linspace(0, 100, n_bins+1))
    # Adjust bounds for precision issues
    bounds[0] -= 1e-8
    bounds[-1] += 1e-8
    bin_calibrations = []
    bin_sizes = []
    for i in range(1, len(bounds)):
        idxs_in_interval = np.logical_and(yhat >= bounds[i-1], yhat < bounds[i])
        assert sum(idxs_in_interval) > 0
        bin_sizes.append(sum(idxs_in_interval))
        bin_calibrations.append(np.abs(np.mean(yhat[idxs_in_interval] - y[idxs_in_interval])))
    assert sum(bin_sizes) == len(yhat)
    
    # Check that the bins are not too uneven; if they are, print sizes
    if max(bin_sizes) - min(bin_sizes) > 0.01*len(yhat):
        print(bin_sizes)
        raise ValueError('Bins are too uneven')

    # Return the average calibration across bins, weighted by bin size
    return np.average(bin_calibrations, weights=bin_sizes)


def calibration_pointplot(
    granular_df,
    coarse_df,
    outcomes = outcomes_all,
    metric = '1_bins',
    ci_bounds = [0.025, 0.975],
    add_coarse_labels = True,
    fontsize_xticks = 14,
    fontsize_yticks = 10,
    fontsize_xlabel = 14,
    fontsize_annot = 14,
    fontsize_title = 16,
    first_annot_only = True,
    annot_x = 0.98,
    suptitle = None,
    fontsize_suptitle = 18,
    filepath = None,
    figsize = None,
    tight_layout = False,
    dpi = 150,
):
    if figsize is None:
        figsize = (len(outcomes)*5, 8)
    f, axs = plt.subplots(nrows=1, ncols=len(outcomes),
                          sharey=True,
                          figsize=figsize, dpi=dpi)

    # make a list if user only wants to plot one outcome
    if len(outcomes) == 1:
        axs = [axs]
        
    for (k, ax) in enumerate(axs):
        outcome = outcomes[k]

        sub_df = pd.DataFrame()
        for granular_race in granular_to_coarse:
            sub_df[granular_race] = granular_df[f'{granular_race}_{outcome}_{metric}']

        # get medians for coarse groups to sort by
        coarse_medians = {}
        for coarse_race in coarse_to_granular.keys():
            coarse_medians[coarse_race] = np.median(coarse_df[f'{coarse_race}_{outcome}_{metric}'])

        if k == 0:
            # sort first by coarse group median, then by granular median
            plot_order = sorted(sub_df.columns, 
                                key = lambda x : tuple([coarse_medians[granular_to_coarse[x]],
                                                        np.median(sub_df[x])]))
            # get sorted coarse races for future use
            sorted_coarse_races = sorted(coarse_medians.keys(), key=lambda x : coarse_medians[x])

        point_colors = []
        for coarse_race in sorted_coarse_races:
            # Fixed color per group
            point_colors += ([coarse_colors[coarse_race]]*len(coarse_to_granular[coarse_race]))

        medians = sub_df[plot_order].median(axis=0).values
        cis = np.array([medians - sub_df[plot_order].quantile(ci_bounds[0]).values, 
                        sub_df[plot_order].quantile(ci_bounds[1]).values - medians])
        ax.scatter(
            x=medians,
            y=sub_df[plot_order].columns,
            alpha=1,
            color=point_colors,
        )
        ax.errorbar(
            x=medians,
            y=sub_df[plot_order].columns,
            xerr=cis,
            fmt='o',
            alpha=1,
            color='none',
            ecolor=point_colors,
        )

        # Draw boundary lines between coarse groups
        coarse_races = np.array([granular_to_coarse[col] for col in plot_order])
        change_idxs = np.where(coarse_races[1:] != coarse_races[:-1])[0] + 1 # get idxs where element i is not equal to element i-1
        for y in change_idxs:
            ax.axhline(y = (y-0.5), 
                       color='#888888', 
                       linestyle='-', 
                       linewidth=0.5,
                       alpha=0.5)
        
        if add_coarse_labels:
            # check if we should only label the first subplot
            if k == 0 or not first_annot_only:
                for c_idx in range(len(sorted_coarse_races)):
                    coarse_race = sorted_coarse_races[c_idx]
                    y_pos = 0 if c_idx == 0 else change_idxs[c_idx-1]
                    ax.text(x=annot_x, 
                            y=1 - (y_pos+0.8) / len(plot_order), 
                            s=coarse_abbrev[coarse_race], 
                            color=coarse_colors[coarse_race],
                            fontsize=fontsize_annot, 
                            ha='right', va='center',
                            transform=ax.transAxes)

        ax.set_ylim([25.5, -0.5])

        # For each coarse group, plot:
        # (1) Dashed line at the median
        # (2) Shaded region for 95% CI. 
        # Note that axvspan goes from 0 (bottom) to 1 (top) in relative coords.
        median_line_alpha = 0.5
        ci_shade_alpha = 0.2
        for (i, coarse_race) in enumerate(sorted_coarse_races):
            ci_low = np.quantile(coarse_df[f'{coarse_race}_{outcome}_{metric}'], ci_bounds[0])
            ci_high = np.quantile(coarse_df[f'{coarse_race}_{outcome}_{metric}'], ci_bounds[1])
            if i == 0:
                ax.vlines(x = coarse_medians[coarse_race],
                           ymin = -0.5,
                           ymax = change_idxs[i] - 0.5,
                           color=coarse_colors[coarse_race], 
                           linestyle='--',
                           alpha=median_line_alpha)
                ax.axvspan(ci_low, ci_high, 
                           ymin = 1 - (change_idxs[i])/len(coarse_races), 
                           ymax = 1, 
                           alpha=ci_shade_alpha, color=coarse_colors[coarse_race])
            elif i == len(change_idxs):
                ax.vlines(x = coarse_medians[coarse_race],
                           ymin = change_idxs[i-1] - 0.5,
                           ymax = len(plot_order) - 0.5,
                           color=coarse_colors[coarse_race], 
                           linestyle='--', 
                           alpha=median_line_alpha)
                ax.axvspan(ci_low, ci_high, 
                           ymin = 0, 
                           ymax = 1-(change_idxs[i-1])/len(coarse_races), 
                           alpha=ci_shade_alpha, color=coarse_colors[coarse_race])
            else:
                ax.vlines(x = coarse_medians[coarse_race],
                           ymin = change_idxs[i-1] - 0.5,
                           ymax = change_idxs[i] - 0.5,
                           color=coarse_colors[coarse_race], 
                           linestyle='--', 
                           alpha=median_line_alpha)
                ax.axvspan(ci_low, ci_high, 
                           ymin = 1-(change_idxs[i])/len(coarse_races), 
                           ymax = 1-(change_idxs[i-1])/len(coarse_races), 
                           alpha=ci_shade_alpha, color=coarse_colors[coarse_race])

        ax.set_title(outcomes_to_labels[outcome], fontsize=fontsize_title)
        if k == 1:
            ax.set_xlabel(metrics_to_labels[metric], fontsize=fontsize_xlabel)
        ax.tick_params(axis='x', which='major', labelsize=fontsize_xticks)
        
        if k == 0:
            ax.tick_params(axis='y', which='major', labelsize=fontsize_yticks)
            new_labels = [granular_abbrev[col] for col in plot_order]
            ax.set_yticklabels(new_labels)
    
    if suptitle:
        plt.suptitle(suptitle, fontsize=fontsize_suptitle, y=0.95)
    
    if tight_layout:
        plt.tight_layout()
        
    plt.show()


def clinical_score_prediction_pointplot(
    granular_df,
    coarse_df,
    risk_score,
    outcomes = outcomes_all,
    metric = 'AUPRC',
    ci_bounds = [0.025, 0.975],
    add_coarse_labels = True,
    fontsize_xticks = 14,
    fontsize_yticks = 10,
    fontsize_xlabel = 14,
    fontsize_annot = 14,
    fontsize_title = 16,
    first_annot_only = True,
    annot_x = 0.98,
    suptitle = None,
    fontsize_suptitle = 18,
    filepath = None,
    figsize = None,
    tight_layout = False,
    dpi = 150,
):
    if figsize is None:
        figsize = (len(outcomes)*5, 8)
    f, axs = plt.subplots(nrows=1, ncols=len(outcomes),
                          sharey=True,
                          figsize=figsize, dpi=dpi)

    # make a list if user only wants to plot one outcome
    if len(outcomes) == 1:
        axs = [axs]
        
    for (k, ax) in enumerate(axs):
        outcome = outcomes[k]

        sub_df = pd.DataFrame()
        for granular_race in granular_to_coarse:
            sub_df[granular_race] = granular_df[f'{risk_score}_{granular_race}_{outcome}_{metric}']

        # get medians for coarse groups to sort by
        coarse_medians = {}
        for coarse_race in coarse_to_granular.keys():
            coarse_medians[coarse_race] = np.median(coarse_df[f'{risk_score}_{coarse_race}_{outcome}_{metric}'])

        if k == 0:
            # sort first by coarse group median, then by granular median
            plot_order = sorted(sub_df.columns, 
                                key = lambda x : tuple([coarse_medians[granular_to_coarse[x]],
                                                        np.median(sub_df[x])]))
            # get sorted coarse races for future use
            sorted_coarse_races = sorted(coarse_medians.keys(), key=lambda x : coarse_medians[x])

        point_colors = []
        for coarse_race in sorted_coarse_races:
            # Fixed color per group
            point_colors += ([coarse_colors[coarse_race]]*len(coarse_to_granular[coarse_race]))

        medians = sub_df[plot_order].median(axis=0).values
        cis = np.array([medians - sub_df[plot_order].quantile(ci_bounds[0]).values, 
                        sub_df[plot_order].quantile(ci_bounds[1]).values - medians])
        ax.scatter(
            x=medians,
            y=sub_df[plot_order].columns,
            alpha=1,
            color=point_colors,
        )
        ax.errorbar(
            x=medians,
            y=sub_df[plot_order].columns,
            xerr=cis,
            fmt='o',
            alpha=1,
            color='none',
            ecolor=point_colors,
        )

        # Draw boundary lines between coarse groups
        coarse_races = np.array([granular_to_coarse[col] for col in plot_order])
        change_idxs = np.where(coarse_races[1:] != coarse_races[:-1])[0] + 1 # get idxs where element i is not equal to element i-1
        for y in change_idxs:
            ax.axhline(y = (y-0.5), 
                       color='#888888', 
                       linestyle='-', 
                       linewidth=0.5,
                       alpha=0.5)
        
        if add_coarse_labels:
            # check if we should only label the first subplot
            if k == 0 or not first_annot_only:
                for c_idx in range(len(sorted_coarse_races)):
                    coarse_race = sorted_coarse_races[c_idx]
                    y_pos = 0 if c_idx == 0 else change_idxs[c_idx-1]
                    ax.text(x=annot_x, 
                            y=1 - (y_pos+0.8) / len(plot_order), 
                            s=coarse_abbrev[coarse_race], 
                            color=coarse_colors[coarse_race],
                            fontsize=fontsize_annot, 
                            ha='right', va='center',
                            transform=ax.transAxes)

        ax.set_ylim([25.5, -0.5])

        # For each coarse group, plot:
        # (1) Dashed line at the median
        # (2) Shaded region for 95% CI. 
        # Note that axvspan goes from 0 (bottom) to 1 (top) in relative coords.
        median_line_alpha = 0.5
        ci_shade_alpha = 0.2
        for (i, coarse_race) in enumerate(sorted_coarse_races):
            ci_low = np.quantile(coarse_df[f'{risk_score}_{coarse_race}_{outcome}_{metric}'], ci_bounds[0])
            ci_high = np.quantile(coarse_df[f'{risk_score}_{coarse_race}_{outcome}_{metric}'], ci_bounds[1])
            if i == 0:
                ax.vlines(x = coarse_medians[coarse_race],
                           ymin = -0.5,
                           ymax = change_idxs[i] - 0.5,
                           color=coarse_colors[coarse_race], 
                           linestyle='--',
                           alpha=median_line_alpha)
                ax.axvspan(ci_low, ci_high, 
                           ymin = 1 - (change_idxs[i])/len(coarse_races), 
                           ymax = 1, 
                           alpha=ci_shade_alpha, color=coarse_colors[coarse_race])
            elif i == len(change_idxs):
                ax.vlines(x = coarse_medians[coarse_race],
                           ymin = change_idxs[i-1] - 0.5,
                           ymax = len(plot_order) - 0.5,
                           color=coarse_colors[coarse_race], 
                           linestyle='--', 
                           alpha=median_line_alpha)
                ax.axvspan(ci_low, ci_high, 
                           ymin = 0, 
                           ymax = 1-(change_idxs[i-1])/len(coarse_races), 
                           alpha=ci_shade_alpha, color=coarse_colors[coarse_race])
            else:
                ax.vlines(x = coarse_medians[coarse_race],
                           ymin = change_idxs[i-1] - 0.5,
                           ymax = change_idxs[i] - 0.5,
                           color=coarse_colors[coarse_race], 
                           linestyle='--', 
                           alpha=median_line_alpha)
                ax.axvspan(ci_low, ci_high, 
                           ymin = 1-(change_idxs[i])/len(coarse_races), 
                           ymax = 1-(change_idxs[i-1])/len(coarse_races), 
                           alpha=ci_shade_alpha, color=coarse_colors[coarse_race])

        ax.set_title(outcomes_to_labels[outcome], fontsize=fontsize_title)
        ax.set_xlabel(metrics_to_labels[metric], fontsize=fontsize_xlabel)
        ax.tick_params(axis='x', which='major', labelsize=fontsize_xticks)
        
        if k == 0:
            ax.tick_params(axis='y', which='major', labelsize=fontsize_yticks)
            new_labels = [granular_abbrev[col] for col in plot_order]
            ax.set_yticklabels(new_labels)
    
    if suptitle:
        plt.suptitle(suptitle, fontsize=fontsize_suptitle, y=0.95)
    
    if tight_layout:
        plt.tight_layout()
        
    plt.show()
    

def outcome_frequency_fixedorder_pointplot(
    granular_df,
    coarse_df,
    outcomes = outcomes_all,
    ci_bounds = [0.025, 0.975],
    xlabel = r"$p$(outcome)",
    add_coarse_labels = False,
    fontsize_xticks = 14,
    fontsize_yticks = 10,
    fontsize_xlabel = 14,
    fontsize_annot = 14,
    fontsize_title = 16,
    first_annot_only = True,
    annot_x = 0.98,
    suptitle = None,
    fontsize_suptitle = 18,
    filepath = None,
    figsize = None,
    tight_layout = True,
    dpi = 150,
):
    if figsize is None:
        figsize = (len(outcomes)*5, 8)
    f, axs = plt.subplots(nrows=1, ncols=len(outcomes),
                          sharey=True,
                          figsize=figsize, dpi=dpi)

    # make a list if user only wants to plot one outcome
    if len(outcomes) == 1:
        axs = [axs]
        
    for (k, ax) in enumerate(axs):
        outcome = outcomes[k]

        sub_df = pd.DataFrame()
        for granular_race in granular_to_coarse:
            sub_df[granular_race] = granular_df[f'{granular_race}_{outcome}']

        # get medians for coarse groups to sort by
        coarse_medians = {}
        for coarse_race in coarse_to_granular.keys():
            coarse_medians[coarse_race] = np.median(coarse_df[f'{coarse_race}_{outcome}'])

        if k == 0:
            # sort first by coarse group median, then by granular median
            plot_order = sorted(sub_df.columns, 
                                key = lambda x : tuple([coarse_medians[granular_to_coarse[x]],
                                                        np.median(sub_df[x])]))
            # get sorted coarse races for future use
            sorted_coarse_races = sorted(coarse_medians.keys(), key=lambda x : coarse_medians[x])

        box_colors = []
        for coarse_race in sorted_coarse_races:
            # Fixed color per group
            box_colors += ([coarse_colors[coarse_race]]*len(coarse_to_granular[coarse_race]))

        medians = sub_df[plot_order].median(axis=0).values
        cis = np.array([medians - sub_df[plot_order].quantile(ci_bounds[0]).values, 
                        sub_df[plot_order].quantile(ci_bounds[1]).values - medians])
        ax.scatter(
            x=medians,
            y=sub_df[plot_order].columns,
            alpha=1,
            color=box_colors,
        )
        ax.errorbar(
            x=medians,
            y=sub_df[plot_order].columns,
            xerr=cis,
            fmt='o',
            alpha=1,
            color='none',
            ecolor=box_colors,
        )

        # Draw boundary lines between coarse groups
        coarse_races = np.array([granular_to_coarse[col] for col in plot_order])
        change_idxs = np.where(coarse_races[1:] != coarse_races[:-1])[0] + 1 # get idxs where element i is not equal to element i-1
        for y in change_idxs:
            ax.axhline(y = (y-0.5), 
                       color='#888888', 
                       linestyle='-', 
                       linewidth=0.5,
                       alpha=0.5)
            
        if add_coarse_labels:
            # check if we should only label the first subplot
            if k == 0 or not first_annot_only:
                for c_idx in range(len(sorted_coarse_races)):
                    coarse_race = sorted_coarse_races[c_idx]
                    y_pos = 0 if c_idx == 0 else change_idxs[c_idx-1]
                    ax.text(x=ax.get_xlim()[1]*annot_x, 
                            y=y_pos+0.4, 
                            s=coarse_abbrev[coarse_race], 
                            color=coarse_colors[coarse_race],
                            fontsize=fontsize_annot, 
                            ha='right', va='center',
                            transform=ax.transData)

        ax.set_ylim([25.5, -0.5])

        # For each coarse group, plot:
        # (1) Dashed line at the median
        # (2) Shaded region for 95% CI. 
        # Note that axvspan goes from 0 (bottom) to 1 (top) in relative coords.
        median_line_alpha = 0.5
        ci_shade_alpha = 0.2
        for (i, coarse_race) in enumerate(sorted_coarse_races):
            ci_low = np.quantile(coarse_df[f'{coarse_race}_{outcome}'], ci_bounds[0])
            ci_high = np.quantile(coarse_df[f'{coarse_race}_{outcome}'], ci_bounds[1])
            if i == 0:
                ax.vlines(x = coarse_medians[coarse_race],
                           ymin = -0.5,
                           ymax = change_idxs[i] - 0.5,
                           color=coarse_colors[coarse_race], 
                           linestyle='--',
                           alpha=median_line_alpha)
                ax.axvspan(ci_low, ci_high, 
                           ymin = 1 - (change_idxs[i])/len(coarse_races), 
                           ymax = 1, 
                           alpha=ci_shade_alpha, color=coarse_colors[coarse_race])
            elif i == len(change_idxs):
                ax.vlines(x = coarse_medians[coarse_race],
                           ymin = change_idxs[i-1] - 0.5,
                           ymax = len(plot_order) - 0.5,
                           color=coarse_colors[coarse_race], 
                           linestyle='--', 
                           alpha=median_line_alpha)
                ax.axvspan(ci_low, ci_high, 
                           ymin = 0, 
                           ymax = 1-(change_idxs[i-1])/len(coarse_races), 
                           alpha=ci_shade_alpha, color=coarse_colors[coarse_race])
            else:
                ax.vlines(x = coarse_medians[coarse_race],
                           ymin = change_idxs[i-1] - 0.5,
                           ymax = change_idxs[i] - 0.5,
                           color=coarse_colors[coarse_race], 
                           linestyle='--', 
                           alpha=median_line_alpha)
                ax.axvspan(ci_low, ci_high, 
                           ymin = 1-(change_idxs[i])/len(coarse_races), 
                           ymax = 1-(change_idxs[i-1])/len(coarse_races), 
                           alpha=ci_shade_alpha, color=coarse_colors[coarse_race])

        ax.set_title(outcomes_to_labels[outcome], fontsize=fontsize_title)
        ax.set_xlabel(xlabel, fontsize=fontsize_xlabel)
        ax.tick_params(axis='x', which='major', labelsize=fontsize_xticks)
        
        if k == 0:
            ax.tick_params(axis='y', which='major', labelsize=fontsize_yticks)
            new_labels = [granular_abbrev[col] for col in plot_order]
            ax.set_yticklabels(new_labels)
            
    if suptitle:
        plt.suptitle(suptitle, fontsize=fontsize_suptitle, y=0.95)   
         
    if tight_layout:
        plt.tight_layout()
        
    plt.show()
    
    
def predictive_metrics_pointplot(
    granular_df,
    coarse_df,
    outcomes = outcomes_all,
    metric = 'AUPRC',
    ci_bounds = [0.025, 0.975],
    add_coarse_labels = True,
    fontsize_xticks = 14,
    fontsize_yticks = 10,
    fontsize_xlabel = 14,
    fontsize_annot = 14,
    fontsize_title = 16,
    first_annot_only = True,
    annot_x = 0.98,
    suptitle = None,
    fontsize_suptitle = 18,
    filepath = None,
    figsize = None,
    tight_layout = False,
    dpi = 150,
):
    if figsize is None:
        figsize = (len(outcomes)*5, 8)
    f, axs = plt.subplots(nrows=1, ncols=len(outcomes),
                          sharey=True,
                          figsize=figsize, dpi=dpi)

    # make a list if user only wants to plot one outcome
    if len(outcomes) == 1:
        axs = [axs]
        
    for (k, ax) in enumerate(axs):
        outcome = outcomes[k]

        sub_df = pd.DataFrame()
        for granular_race in granular_to_coarse:
            sub_df[granular_race] = granular_df[f'{granular_race}_{outcome}_{metric}']

        # get medians for coarse groups to sort by
        coarse_medians = {}
        for coarse_race in coarse_to_granular.keys():
            coarse_medians[coarse_race] = np.median(coarse_df[f'{coarse_race}_{outcome}_{metric}'])

        if k == 0:
            # sort first by coarse group median, then by granular median
            plot_order = sorted(sub_df.columns, 
                                key = lambda x : tuple([coarse_medians[granular_to_coarse[x]],
                                                        np.median(sub_df[x])]))
            # get sorted coarse races for future use
            sorted_coarse_races = sorted(coarse_medians.keys(), key=lambda x : coarse_medians[x])

        point_colors = []
        for coarse_race in sorted_coarse_races:
            # Fixed color per group
            point_colors += ([coarse_colors[coarse_race]]*len(coarse_to_granular[coarse_race]))

        medians = sub_df[plot_order].median(axis=0).values
        cis = np.array([medians - sub_df[plot_order].quantile(ci_bounds[0]).values, 
                        sub_df[plot_order].quantile(ci_bounds[1]).values - medians])
        ax.scatter(
            x=medians,
            y=sub_df[plot_order].columns,
            alpha=1,
            color=point_colors,
        )
        ax.errorbar(
            x=medians,
            y=sub_df[plot_order].columns,
            xerr=cis,
            fmt='o',
            alpha=1,
            color='none',
            ecolor=point_colors,
        )

        # Draw boundary lines between coarse groups
        coarse_races = np.array([granular_to_coarse[col] for col in plot_order])
        change_idxs = np.where(coarse_races[1:] != coarse_races[:-1])[0] + 1 # get idxs where element i is not equal to element i-1
        for y in change_idxs:
            ax.axhline(y = (y-0.5), 
                       color='#888888', 
                       linestyle='-', 
                       linewidth=0.5,
                       alpha=0.5)
        
        if add_coarse_labels:
            # check if we should only label the first subplot
            if k == 0 or not first_annot_only:
                for c_idx in range(len(sorted_coarse_races)):
                    coarse_race = sorted_coarse_races[c_idx]
                    y_pos = 0 if c_idx == 0 else change_idxs[c_idx-1]
                    ax.text(x=annot_x, 
                            y=1 - (y_pos+0.8) / len(plot_order), 
                            s=coarse_abbrev[coarse_race], 
                            color=coarse_colors[coarse_race],
                            fontsize=fontsize_annot, 
                            ha='right', va='center',
                            transform=ax.transAxes)

        ax.set_ylim([25.5, -0.5])

        # For each coarse group, plot:
        # (1) Dashed line at the median
        # (2) Shaded region for 95% CI. 
        # Note that axvspan goes from 0 (bottom) to 1 (top) in relative coords.
        median_line_alpha = 0.5
        ci_shade_alpha = 0.2
        for (i, coarse_race) in enumerate(sorted_coarse_races):
            ci_low = np.quantile(coarse_df[f'{coarse_race}_{outcome}_{metric}'], ci_bounds[0])
            ci_high = np.quantile(coarse_df[f'{coarse_race}_{outcome}_{metric}'], ci_bounds[1])
            if i == 0:
                ax.vlines(x = coarse_medians[coarse_race],
                           ymin = -0.5,
                           ymax = change_idxs[i] - 0.5,
                           color=coarse_colors[coarse_race], 
                           linestyle='--',
                           alpha=median_line_alpha)
                ax.axvspan(ci_low, ci_high, 
                           ymin = 1 - (change_idxs[i])/len(coarse_races), 
                           ymax = 1, 
                           alpha=ci_shade_alpha, color=coarse_colors[coarse_race])
            elif i == len(change_idxs):
                ax.vlines(x = coarse_medians[coarse_race],
                           ymin = change_idxs[i-1] - 0.5,
                           ymax = len(plot_order) - 0.5,
                           color=coarse_colors[coarse_race], 
                           linestyle='--', 
                           alpha=median_line_alpha)
                ax.axvspan(ci_low, ci_high, 
                           ymin = 0, 
                           ymax = 1-(change_idxs[i-1])/len(coarse_races), 
                           alpha=ci_shade_alpha, color=coarse_colors[coarse_race])
            else:
                ax.vlines(x = coarse_medians[coarse_race],
                           ymin = change_idxs[i-1] - 0.5,
                           ymax = change_idxs[i] - 0.5,
                           color=coarse_colors[coarse_race], 
                           linestyle='--', 
                           alpha=median_line_alpha)
                ax.axvspan(ci_low, ci_high, 
                           ymin = 1-(change_idxs[i])/len(coarse_races), 
                           ymax = 1-(change_idxs[i-1])/len(coarse_races), 
                           alpha=ci_shade_alpha, color=coarse_colors[coarse_race])

        ax.set_title(outcomes_to_labels[outcome], fontsize=fontsize_title)
        ax.set_xlabel(metrics_to_labels[metric], fontsize=fontsize_xlabel)
        ax.tick_params(axis='x', which='major', labelsize=fontsize_xticks)
        
        if k == 0:
            ax.tick_params(axis='y', which='major', labelsize=fontsize_yticks)
            new_labels = [granular_abbrev[col] for col in plot_order]
            ax.set_yticklabels(new_labels)
            
    if suptitle:
        plt.suptitle(suptitle, fontsize=fontsize_suptitle, y=0.95)
            
    if tight_layout:
        plt.tight_layout()
        
    plt.show()
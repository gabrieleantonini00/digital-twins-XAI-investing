"""Generate LaTeX tables and findings only from the completed study and R outputs."""
import hashlib
import json
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER = ROOT / 'paper'


def tex(text):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(c, c) for c in str(text))


def pformat(value):
    if value is None:
        return 'not estimable'
    return r'$p<.001$' if value < .001 else f'$p={value:.3f}$'


def table(headers, rows, columns, caption, note=None):
    output = [r'\begin{table}[htbp]', r'\centering', r'\small', '\\caption{' + caption + '}',
              '\\begin{tabular}{' + columns + '}', r'\toprule', ' & '.join(headers) + r'\\', r'\midrule']
    output += [' & '.join(map(str, row)) + r'\\' for row in rows]
    output += [r'\bottomrule', r'\end{tabular}']
    if note:
        output += [r'\par\smallskip', r'\begin{minipage}{.96\linewidth}\footnotesize ' + note + r'\end{minipage}']
    output += [r'\end{table}']
    return '\n'.join(output)


def main():
    audit = json.loads((ROOT / 'study_audit.json').read_text())
    report = json.loads((ROOT / 'results/report_data.json').read_text())
    config = json.loads((ROOT / 'study_config.json').read_text())
    checksum = hashlib.sha256((ROOT / 'Data_500.csv').read_bytes()).hexdigest()
    if (audit['data_sha256'] != checksum or report['n'] != 500 or audit['n'] != 500
            or report.get('input_data_md5') != hashlib.md5((ROOT / 'Data_500.csv').read_bytes()).hexdigest()):
        raise ValueError('Results and study audit must refer to the completed 500-persona dataset')
    chi, df, p = report['primary_chisq'], report['primary_df'], report['primary_p']
    diag_path = ROOT / 'results/proportional_odds_diagnostics.csv'
    diags = list(csv.DictReader(diag_path.open(encoding='utf-8'))) if diag_path.exists() else []
    failed = [r for r in diags if r.get('Pr(>Chi)') not in ('NA', '', None) and float(r['Pr(>Chi)']) < .05]
    primary = f'$\\chi^2({df})={chi:.2f}$, {pformat(p)}'
    if p < .05:
        abstract = f'The primary three-way interaction was statistically detectable ({primary}), indicating joint moderation within this simulation.'
        conclusion = ('The expanded experiment detected a joint interaction of market context, disclosure, and the inferred literacy proxy. '
                      'Its conditional estimates should be interpreted with the model diagnostics and measurement limitations, and evaluated in a human replication.')
    else:
        abstract = f'The primary three-way interaction was not statistically detectable ({primary}), providing no clear support for joint moderation in this run.'
        conclusion = ('The expanded experiment did not provide clear statistical support for the proposed joint moderation of explanation effects by market and the inferred literacy proxy. '
                      'This result does not establish that moderation is absent. The conditional estimates and their uncertainty define the evidence available from this simulation.')
    if failed:
        abstract += ' Proportional-odds diagnostics flagged departures involving literacy, limiting interpretation of the common odds ratios.'
    (PAPER / 'results.tex').write_text('\\newcommand{\\AbstractFinding}{' + abstract + '}\n\\newcommand{\\ConclusionFinding}{' + conclusion + '}\n', encoding='utf-8')

    desc = report['descriptives']
    body = [r'\subsection{Sample and descriptive distributions}',
            'All 500 sampled personas produced valid baseline and outcome responses. Each randomized condition contains 125 outcomes. '
            f'The median inferred financial-literacy score was {report["median_literacy"]:g}. '
            'The distribution of the ordinal outcome is displayed in Figure~\\ref{fig:distribution}.']
    missing = report.get('missing_demographics', {})
    if missing:
        body.append('Missing demographic fields were retained rather than imputed: ' + ', '.join(f'{tex(k.lower())} ({v})' for k, v in missing.items()) + '. These fields were not required by the prespecified primary model.')
    body.append(table(['Market', 'Disclosure', '$N$', 'Mean', 'SD', 'Median'],
                      [[r['Market'], r['Transparency_F'], r['N'], f'{r["Mean"]:.2f}', f'{r["SD"]:.2f}', f'{r["Median"]:g}'] for r in desc],
                      'llrrrr', 'Descriptive willingness to invest by randomized condition',
                      'Means summarize the coded 1--7 scores. The primary model treats the outcome as ordinal.'))
    body += [r'\begin{figure}[htbp]\centering', r'\includegraphics[width=\linewidth]{../results/outcome_distributions.png}',
             r'\caption{Distribution of single simulated investment responses by market and disclosure.}\label{fig:distribution}', r'\end{figure}',
             r'\subsection{Primary model and planned comparisons}',
             f'The likelihood-ratio test of the three-way interaction yielded {primary}. ' +
             ('This supports a joint interaction within the fitted model.' if p < .05 else
              'The new data therefore do not provide clear statistical support for the joint-moderation expectation. A nonsignificant test is not evidence of equivalence or of a zero interaction.'),
             f'The maximum absolute gradient was {report["ordinal_max_gradient"]:.2e}; the Hessian condition number was {report["ordinal_condition_number"]:.1f}. '
             'Coefficient intervals in Table~\\ref{tab:coefficients} are pointwise Wald intervals.']
    names = {
        'MarketBull': 'Bull market', 'Transparency_FXAI': 'XAI disclosure', 'Literacy_c': 'Literacy (centered)',
        'MarketBull:Transparency_FXAI': 'Market $\\times$ XAI', 'MarketBull:Literacy_c': 'Market $\\times$ Literacy',
        'Transparency_FXAI:Literacy_c': 'XAI $\\times$ Literacy',
        'MarketBull:Transparency_FXAI:Literacy_c': 'Market $\\times$ XAI $\\times$ Literacy',
    }
    coeff_table = table(['Term', '$\\beta$', 'SE', 'OR', '95\\% CI', '$p$'],
                        [[names.get(r['Term'], tex(r['Term'])), f'{r["Estimate"]:.3f}', f'{r["SE"]:.3f}', f'{r["OR"]:.2f}',
                          f'[{r["OR_Lower"]:.2f}, {r["OR_Upper"]:.2f}]', '$<.001$' if r['p'] < .001 else f'{r["p"]:.3f}']
                         for r in report['ordinal_coefficients']],
                        'lrrrrr', 'Cumulative logit location coefficients\\label{tab:coefficients}',
                        'Reference conditions: bear market and standard AI. Literacy is centered at 4. With interactions, lower-order coefficients are conditional effects at the reference values. ORs for interaction terms are ratios of odds ratios.')
    body.append(coeff_table)
    contrasts = report['planned_contrasts']
    body.append(table(['Market', 'Literacy', 'OR (XAI/Std.)', '95\\% CI', '$p_{\\mathrm{Holm}}$', 'Extrap.'],
                      [[r['Market'], r['Literacy'], f'{r["OR"]:.2f}', f'[{r["OR_Lower"]:.2f}, {r["OR_Upper"]:.2f}]',
                        '$<.001$' if r['p_Holm'] < .001 else f'{r["p_Holm"]:.3f}', 'Yes' if r['Extrapolation'] else 'No'] for r in contrasts],
                      'lrrrrl', 'Planned XAI versus standard-AI contrasts',
                      'Holm correction covers all six tests. Confidence intervals are pointwise, without multiplicity adjustment. Extrapolation means that the proxy value is outside the observed range in at least one compared treatment cell.'))
    supported = [r for r in contrasts if r['p_Holm'] < .05 and not r['Extrapolation']]
    body.append(f'{len(supported)} of the six planned contrasts were statistically detectable after Holm correction and within the observed cell ranges. '
                'The direction and magnitude of each comparison are shown in the table; significance differences across subgroups do not themselves establish an interaction.')
    if diag_path.exists():
        if failed:
            body.append('The nominal-effects diagnostics flagged departures from proportional odds for ' + str(len(failed)) +
                        ' candidate terms at the unadjusted .05 level. Consequently, the common-odds-ratio interpretation is approximate. '
                        'These diagnostic tests can be unstable in sparse categories and were not used to select a replacement primary model. Full diagnostics accompany the replication files.')
        else:
            body.append('The available nominal-effects diagnostics did not flag an unadjusted $p<.05$ departure from proportional odds. This does not prove that the assumption holds; full diagnostics accompany the replication files.')
        unavailable = [r for r in diags if r.get('') not in ('<none>', '', None) and r.get('Pr(>Chi)') in ('NA', '', None)]
        if unavailable:
            body.append(f'{len(unavailable)} diagnostic comparisons could not be estimated. These unavailable tests provide no evidence for or against proportional odds.')
    else:
        body.append('Proportional-odds diagnostics could not be estimated; the analysis log records the failure. This limits interpretation of the common odds ratios.')

    body.append(r'\subsection{Sensitivity analyses}')
    ols = next(r for r in report['ols_coefficients'] if r['Term'] == 'MarketBull:Transparency_FXAI:Literacy_c')
    body.append(f'The corresponding OLS three-way coefficient was {ols["Estimate"]:.3f} (HC3 SE = {ols["SE_HC3"]:.3f}; {pformat(ols["p"])}). '
                'This is a sensitivity check on the coded score scale, not an ordinal-model estimate.')
    art = report.get('art_anova')
    if art:
        triple = next(r for r in art if r['Term'] == 'Market:Transparency_F:Literacy_Group')
        body.append(f'The median-split ART sensitivity test yielded $F({triple["Df"]},{triple["Df.res"]})={triple["F value"]:.2f}$, {pformat(triple["Pr(>F)"])}. '
                    'The low/high literacy group sizes are reported below; no balanced allocation was imposed on this nonrandomized moderator.')
        body.append(table(['Market', 'Disclosure', 'Literacy group', '$N$', 'Mean', 'SD'],
                          [[r['Market'], r['Transparency_F'], r['Literacy_Group'], r['N'], f'{r["Mean"]:.2f}', f'{r["SD"]:.2f}'] for r in report['group_descriptives']],
                          'lllrrr', 'Descriptive median-split subgroups', 'Low: literacy proxy at or below the sample median; High: above the median.'))
        body += [r'\begin{figure}[htbp]\centering', r'\includegraphics[width=\linewidth]{../results/interaction_descriptive.png}',
                 r'\caption{Descriptive means for median-split literacy groups. The primary model retains numerical literacy.}', r'\end{figure}']
    else:
        body.append('The ART sensitivity analysis was unavailable because at least one median-split subgroup was empty.')
    (PAPER / 'results_body.tex').write_text('\n\n'.join(body) + '\n', encoding='utf-8')

    discussion = [abstract,
                  'The planned contrasts describe where the fitted model places differences between the two disclosure descriptions. '
                  'They do not identify emotional mediation, observed trust, or investment performance. Nor does a significant comparison in one market and a nonsignificant comparison in the other demonstrate that the two effects differ.',
                  'Baseline separation makes the inferred literacy characterization independent of exposure to the assigned vignette. '
                  'However, the outcome request explicitly includes that characterization, so any moderation can partly reflect the model responding to its own prior labels. '
                  'A human study with independently measured financial knowledge is needed to evaluate whether the simulated heterogeneity corresponds to investor behavior.',
                  'The observations represent one sampled response from each distinct persona under one fixed model configuration. '
                  'The experiment was not designed to estimate within-person response variability or compare temperatures. '
                  'The inferential uncertainty reported here is conditional on the chosen profiles, prompts, model, and configuration.']
    (PAPER / 'discussion.tex').write_text('\n\n'.join(discussion) + '\n', encoding='utf-8')
    receipts = audit['receipts']
    inputs = sum(r['input_tokens'] for r in receipts)
    outputs = sum(r['output_tokens'] for r in receipts)
    reasoning = sum(r['reasoning_tokens'] for r in receipts)
    cost = audit['total_estimated_actual_cost_usd']
    audit_tex = [f'The completed study used {inputs:,} input tokens and {outputs:,} output tokens, including {reasoning:,} reasoning tokens. '
                 f'The estimated API charge from recorded usage and published Batch rates was \\${cost:.2f}, before taxes or provider adjustments. '
                 'The no-persona technical compatibility check is excluded from this study total. No chargeable request was processed in the initially rejected 500-request batch.',
                 'The source revision was {\\footnotesize\\texttt{' + tex(config['dataset_revision']) + '}}. '
                 'The final dataset SHA-256 checksum is recorded in the public study audit. The model settings follow the selected model documentation \\citep{openaiModel}.',
                 table(['Stage', 'Valid personas', 'Input tokens', 'Output tokens', 'Est. USD'],
                       [[r['stage'].capitalize(), r['valid_n'], f'{r["input_tokens"]:,}', f'{r["output_tokens"]:,}', f'{r["estimated_actual_cost_usd"]:.2f}'] for r in receipts],
                       'lrrrr', 'Recorded Batch usage', 'Output tokens include reasoning. Estimated costs include reported cache reads and writes. Each stage used three sequential transport batches.')]
    (PAPER / 'audit.tex').write_text('\n\n'.join(audit_tex) + '\n', encoding='utf-8')
    print('Generated manuscript findings and tables from the completed, checksum-verified study.')


if __name__ == '__main__':
    main()

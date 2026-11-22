"""Charts of prediction probabilities and already published evaluation counts."""
import altair as alt
import pandas as pd


def probability_chart(result):
    rows = [{'Label': label, 'Series': series, 'Probability': result[field][label]}
            for label in result['class_order']
            for series, field in [('Raw', 'raw_probabilities'),
                                  ('Calibrated', 'calibrated_probabilities')]]
    bars = alt.Chart(pd.DataFrame(rows)).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X('Probability:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='.0%')),
        y=alt.Y('Label:N', sort=result['class_order'], title=None),
        yOffset=alt.YOffset('Series:N', sort=['Raw', 'Calibrated']),
        color=alt.Color('Series:N', scale=alt.Scale(domain=['Raw', 'Calibrated'],
                                                   range=['#a3c7c0', '#176b60']),
                        legend=alt.Legend(orient='top', title=None)),
        tooltip=['Label:N', 'Series:N', alt.Tooltip('Probability:Q', format='.2%')])
    threshold = alt.Chart(pd.DataFrame({'Threshold': [result.get('threshold', .5)]})).mark_rule(
        color='#9b5b20', strokeDash=[5, 4]).encode(x='Threshold:Q')
    return (bars + threshold).properties(height=360).configure_view(stroke=None)


def confusion_rows(counts):
    """Derive integer patient counts, never from probabilities or unlabeled uploads."""
    positive, negative, fn, predicted = [counts[k] for k in
                                       ('positive', 'negative', 'false_negative', 'predicted_positive')]
    if any(type(v) is not int or v < 0 for v in (positive, negative, fn, predicted)):
        raise ValueError('Expected nonnegative integer patient counts')
    tp = positive - fn
    fp = predicted - tp
    tn = negative - fp
    if min(tp, fp, tn) < 0:
        raise ValueError('Inconsistent published patient counts')
    return [{'Actual': a, 'Predicted': p, 'Outcome': outcome, 'Patients': n}
            for a, p, outcome, n in [('Absent', 'Absent', 'TN', tn),
                                     ('Absent', 'Present', 'FP', fp),
                                     ('Present', 'Absent', 'FN', fn),
                                     ('Present', 'Present', 'TP', tp)]]


def confusion_chart(rows):
    base = alt.Chart(pd.DataFrame(rows)).encode(
        x=alt.X('Predicted:N', sort=['Absent', 'Present'], title='Predicted label'),
        y=alt.Y('Actual:N', sort=['Absent', 'Present'], title='Recorded reference label'),
        tooltip=['Actual:N', 'Predicted:N', 'Outcome:N', 'Patients:Q'])
    cells = base.mark_rect(cornerRadius=5).encode(
        color=alt.Color('Patients:Q', scale=alt.Scale(scheme='tealblues'), legend=None))
    text = base.mark_text(fontSize=24, fontWeight='bold', stroke='white', strokeWidth=.3).encode(
        text='Patients:Q', color=alt.value('#102f35'))
    return (cells + text).properties(height=260).configure_view(stroke=None)

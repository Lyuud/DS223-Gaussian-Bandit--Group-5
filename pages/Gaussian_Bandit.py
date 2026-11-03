import pandas as pd
import streamlit as st
from gaussian_bandit import run_gaussian_bandit

st.title('Gaussian Bandit — Thompson Sampling')
st.write('Continuous Normal rewards, known noise σ, and Normal(0, 25) prior for each unknown mean.')
with st.form('gaussian_inputs'):
    c1, c2, c3 = st.columns(3)
    mu1 = c1.number_input('True mean — Alternative 1', value=1.0)
    mu2 = c2.number_input('True mean — Alternative 2', value=2.0)
    mu3 = c3.number_input('True mean — Alternative 3', value=3.0)
    c4, c5, c6 = st.columns(3)
    sigma = c4.number_input('Known noise σ', value=1.0, min_value=0.000001, format='%.6f')
    rounds = c5.number_input('Number of rounds', min_value=1, value=100, step=1)
    seed = c6.number_input('Random seed', min_value=0, value=42, step=1)
    submitted = st.form_submit_button('Run Gaussian Thompson Sampling', type='primary')
if submitted:
    if sigma <= 0 or rounds <= 0:
        st.error('σ and rounds must be positive.')
    else:
        st.session_state['gaussian_result'] = run_gaussian_bandit([mu1, mu2, mu3], float(sigma), int(rounds), int(seed))
if 'gaussian_result' in st.session_state:
    r = st.session_state['gaussian_result']
    st.subheader('Simulation results')
    a, b, c = st.columns(3)
    a.metric('Rounds', r['rounds'])
    b.metric('Total observed reward', f"{r['total_reward']:.3f}")
    c.metric('Final recommendation', f"Alternative {r['recommendation']}")
    df = pd.DataFrame({'Alternative': ['Alternative 1', 'Alternative 2', 'Alternative 3'],
                       'True mean': r['true_means'], 'Posterior estimated mean': r['posterior_means'],
                       'Posterior standard deviation': [v**0.5 for v in r['posterior_variances']],
                       'Selection count': r['counts']})
    st.dataframe(df, hide_index=True, width='stretch')
    st.write(f"Selection count check: {sum(r['counts'])} / {r['rounds']}")
    st.subheader('Posterior mean evolution')
    history = pd.DataFrame(r['trajectories'], columns=['Alternative 1','Alternative 2','Alternative 3'])
    history.index += 1
    st.line_chart(history)
    st.subheader('Selection counts')
    st.bar_chart(df.set_index('Alternative')['Selection count'])
    with st.expander('Observed continuous rewards (including negative values)'):
        st.dataframe(pd.DataFrame({'Round': range(1,r['rounds']+1), 'Alternative':r['chosen'], 'Reward':r['rewards']}),hide_index=True)

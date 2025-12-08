"""
AdaptiveChain - Interactive Supply Chain RL Dashboard
Streamlit application for visualizing and comparing RL agents
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import os

# Add src to path FIRST
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all project modules at startup (CRITICAL FIX)
from src.environment.supply_chain_env import SupplyChainEnv
from src.environment.multi_warehouse_env import MultiWarehouseEnv
from src.environment.env_wrappers import FlattenMultiDiscreteWrapper
from src.agents.baseline_policies import create_baseline_policies
from src.agents.independent_agents import IndependentMultiAgent
from src.agents.coordinated_agents import CoordinatedMultiAgent
from stable_baselines3 import DQN

# Page configuration
st.set_page_config(
    page_title="AdaptiveChain - Supply Chain RL",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - IMPROVED CONTRAST
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
    }
    .insight-box {
        background-color: #fff9e6;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border: 2px solid #f39c12;
        border-left: 6px solid #f39c12;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
    }
    .insight-box h4 {
        color: #1a1a1a;
        font-weight: 700;
        margin-top: 0;
        font-size: 1.4rem;
        margin-bottom: 1rem;
    }
    .insight-box ul {
        color: #1a1a1a;
        font-size: 1.05rem;
        line-height: 1.9;
        margin-left: 1.2rem;
    }
    .insight-box li {
        margin-bottom: 0.7rem;
        color: #2c3e50;
    }
    .insight-box b {
        color: #000000;
        font-weight: 700;
    }
    .insight-box p {
        color: #1a1a1a;
        font-size: 1.05rem;
        line-height: 1.7;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_all_results():
    """Load all saved results"""
    results = {}
    
    try:
        with open('data/baseline_results.json', 'r') as f:
            results['baselines'] = json.load(f)
    except:
        results['baselines'] = {}
    
    try:
        with open('models/dqn_optimized_results.json', 'r') as f:
            results['dqn'] = json.load(f)
    except:
        results['dqn'] = {}
    
    try:
        with open('data/multi_agent_results.json', 'r') as f:
            results['multi_agent'] = json.load(f)
    except:
        results['multi_agent'] = {}
    
    try:
        with open('data/scenario_testing_results.json', 'r') as f:
            results['scenarios'] = json.load(f)
    except:
        results['scenarios'] = {}
    
    try:
        with open('data/ablation_study_results.json', 'r') as f:
            results['ablation'] = json.load(f)
    except:
        results['ablation'] = {}
    
    return results


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">📦 AdaptiveChain</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #34495e; font-weight: 500;">Multi-Agent Reinforcement Learning for Supply Chain Optimization</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("**Final Project: INFO 7375**")
    st.sidebar.markdown("*Reinforcement Learning for Agentic AI*")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Select Page:",
        ["📊 Overview", "📈 Learning Curves", "🎮 Live Simulation",
         "🧪 Scenario Analysis", "🔍 Ablation Study", "📋 Technical Details"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("**Project by:** Sravan Kumar Kurapati\n\n**Course:** INFO 7375\n\n**Institution:** Northeastern University")
    
    # Load results
    results = load_all_results()
    
    # Route to pages
    if page == "📊 Overview":
        show_overview_page(results)
    elif page == "📈 Learning Curves":
        show_learning_curves_page(results)
    elif page == "🎮 Live Simulation":
        show_simulation_page(results)
    elif page == "🧪 Scenario Analysis":
        show_scenario_page(results)
    elif page == "🔍 Ablation Study":
        show_ablation_page(results)
    elif page == "📋 Technical Details":
        show_technical_page(results)


def show_overview_page(results):
    """Overview page"""
    
    st.header("📊 Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="🎯 Best Overall", value="Reorder Point", delta="$1.06M")
    
    with col2:
        st.metric(label="🤖 Best RL Agent", value="DQN Single", delta="$2.27M")
    
    with col3:
        if results.get('multi_agent'):
            improvement = results['multi_agent'].get('coordination_improvement', 0)
            st.metric(
                label="🔄 Coordination Impact",
                value=f"{abs(improvement):.1f}%",
                delta="Worse" if improvement < 0 else "Better",
                delta_color="inverse" if improvement < 0 else "normal"
            )
    
    with col4:
        st.metric(label="🧪 Scenarios Tested", value="5", delta="Comprehensive")
    
    st.markdown("---")
    st.subheader("📊 Complete Performance Comparison")
    
    chart_data = []
    
    if results.get('baselines'):
        b = results['baselines']
        chart_data.extend([
            {'Policy': 'Random (1 WH)', 'Cost': b['random']['mean_cost'], 'Type': 'Baseline'},
            {'Policy': 'Reorder Point (1 WH)', 'Cost': b['reorder_point']['mean_cost'], 'Type': 'Baseline'},
            {'Policy': 'EOQ (1 WH)', 'Cost': b['eoq']['mean_cost'], 'Type': 'Baseline'},
            {'Policy': 'Random (3 WH)', 'Cost': b['random']['mean_cost'] * 3, 'Type': 'Baseline'},
            {'Policy': 'Reorder Point (3 WH)', 'Cost': b['reorder_point']['mean_cost'] * 3, 'Type': 'Baseline'}
        ])
    
    if results.get('dqn'):
        chart_data.append({'Policy': 'DQN (1 WH)', 'Cost': results['dqn']['mean_cost'], 'Type': 'RL Agent'})
    
    if results.get('multi_agent'):
        ma = results['multi_agent']
        chart_data.extend([
            {'Policy': 'Independent DQN (3 WH)', 'Cost': ma['independent']['mean_cost'], 'Type': 'Multi-Agent'},
            {'Policy': 'Coordinated DQN (3 WH)', 'Cost': ma['coordinated']['mean_cost'], 'Type': 'Multi-Agent'}
        ])
    
    df_chart = pd.DataFrame(chart_data)
    
    fig = px.bar(
        df_chart, x='Policy', y='Cost', color='Type',
        title='All Policies Comparison',
        labels={'Cost': 'Total Cost ($)', 'Policy': 'Policy Type'},
        color_discrete_map={'Baseline': '#3498db', 'RL Agent': '#9b59b6', 'Multi-Agent': '#e74c3c'},
        text='Cost'
    )
    
    fig.update_traces(texttemplate='$%{text:.2s}', textposition='outside')
    fig.update_layout(height=500, showlegend=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🔍 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4>✅ What Worked</h4>
        <ul>
        <li><b>DQN Learning:</b> 57% improvement over random ($5.3M → $2.3M)</li>
        <li><b>Classical OR:</b> Reorder Point optimal for stable demand ($1.06M)</li>
        <li><b>Ablation Study:</b> Transfers provide 10.3% benefit</li>
        <li><b>Robustness:</b> Tested across 5 disruption scenarios</li>
        <li><b>Statistical Rigor:</b> All differences significant (p < 0.001)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box" style="border-color: #e74c3c; background-color: #ffe6e6;">
        <h4 style="color: #c0392b;">⚠️ Critical Discovery</h4>
        <ul>
        <li><b>Coordination Paradox:</b> Coordinated 133% worse ($12.9M vs $5.5M)</li>
        <li><b>Root Cause:</b> Excessive transfers (326 vs 51 per episode)</li>
        <li><b>Transfer Overhead:</b> Costs exceeded stockout prevention</li>
        <li><b>Lesson:</b> Coordination needs cost-benefit analysis</li>
        <li><b>Recommendation:</b> Information sharing, not physical transfers</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📋 Detailed Results")
    
    if not df_chart.empty:
        df_display = df_chart.copy()
        df_display['Cost'] = df_display['Cost'].apply(lambda x: f"${x:,.0f}")
        
        if results.get('baselines'):
            random_cost = results['baselines']['random']['mean_cost']
            random_3wh = random_cost * 3
            df_display['vs Random'] = df_chart.apply(
                lambda row: f"{((random_cost if '1 WH' in row['Policy'] else random_3wh) - row['Cost']) / (random_cost if '1 WH' in row['Policy'] else random_3wh) * 100:.1f}%",
                axis=1
            )
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)


def run_baseline_simulation(env, policy, episode_length):
    """Run baseline policy simulation"""
    obs, info = env.reset()
    policy.reset()
    
    history = {
        'days': [],
        'inventory': {'PROD_A': [], 'PROD_B': [], 'PROD_C': []},
        'costs': [],
        'cumulative_cost': []
    }
    
    cumulative = 0
    
    for day in range(episode_length):
        history['days'].append(day)
        
        for i, sku in enumerate(['PROD_A', 'PROD_B', 'PROD_C']):
            history['inventory'][sku].append(info['inventory'].get(sku, 0))
        
        action = policy.select_action(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        
        step_cost = -reward
        cumulative += step_cost
        history['costs'].append(step_cost)
        history['cumulative_cost'].append(cumulative)
        
        if terminated or truncated:
            break
    
    return history


def run_dqn_simulation(env, model, episode_length):
    """Run DQN simulation"""
    obs, info = env.reset()
    
    history = {
        'days': [],
        'inventory': {'PROD_A': [], 'PROD_B': [], 'PROD_C': []},
        'costs': [],
        'cumulative_cost': []
    }
    
    cumulative = 0
    
    for day in range(episode_length):
        history['days'].append(day)
        
        unwrapped_info = env.unwrapped._get_info()
        for sku in ['PROD_A', 'PROD_B', 'PROD_C']:
            history['inventory'][sku].append(unwrapped_info['inventory'].get(sku, 0))
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        step_cost = -reward
        cumulative += step_cost
        history['costs'].append(step_cost)
        history['cumulative_cost'].append(cumulative)
        
        if terminated or truncated:
            break
    
    return history


def run_multiagent_simulation(env, agent, episode_length):
    """Run multi-agent simulation"""
    obs, info = env.reset()
    
    history = {
        'days': [],
        'warehouses': {f'WH_{i}': {'PROD_A': [], 'PROD_B': [], 'PROD_C': []} for i in range(3)},
        'costs': [],
        'cumulative_cost': [],
        'transfers': []
    }
    
    cumulative = 0
    transfer_count = 0
    
    for day in range(episode_length):
        history['days'].append(day)
        
        for wh_idx, wh_info in enumerate(info['warehouses']):
            for sku in ['PROD_A', 'PROD_B', 'PROD_C']:
                history['warehouses'][f'WH_{wh_idx}'][sku].append(wh_info['inventory'].get(sku, 0))
        
        transfer_count = info.get('transfer_count', transfer_count)
        history['transfers'].append(transfer_count)
        
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        step_cost = -reward
        cumulative += step_cost
        history['costs'].append(step_cost)
        history['cumulative_cost'].append(cumulative)
        
        if terminated or truncated:
            break
    
    return history


def display_simulation_results(history, agent_name, episode_length):
    """Display simulation results"""
    
    st.success(f"✅ Simulation Complete! {len(history['days'])} days executed")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_cost = history['cumulative_cost'][-1] if history['cumulative_cost'] else 0
        st.metric("Total Cost", f"${total_cost:,.0f}")
    
    with col2:
        avg_daily = total_cost / len(history['days']) if history['days'] else 0
        st.metric("Avg Daily Cost", f"${avg_daily:,.0f}")
    
    with col3:
        if 'transfers' in history and history['transfers']:
            total_transfers = history['transfers'][-1]
            st.metric("Total Transfers", f"{total_transfers}")
        else:
            st.metric("Agent Type", "Single Warehouse")
    
    st.markdown("---")
    st.subheader("💰 Cumulative Cost Over Time")
    
    fig_cost = go.Figure()
    fig_cost.add_trace(go.Scatter(
        x=history['days'],
        y=history['cumulative_cost'],
        mode='lines',
        name='Cumulative Cost',
        line=dict(color='#e74c3c', width=3),
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.2)'
    ))
    
    fig_cost.update_layout(
        xaxis_title="Day",
        yaxis_title="Cumulative Cost ($)",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_cost, use_container_width=True)
    
    st.subheader("📦 Inventory Levels Over Time")
    
    if 'inventory' in history:
        fig_inv = go.Figure()
        colors = {'PROD_A': '#3498db', 'PROD_B': '#2ecc71', 'PROD_C': '#9b59b6'}
        
        for sku in ['PROD_A', 'PROD_B', 'PROD_C']:
            if sku in history['inventory']:
                fig_inv.add_trace(go.Scatter(
                    x=history['days'],
                    y=history['inventory'][sku],
                    mode='lines',
                    name=sku,
                    line=dict(width=2, color=colors[sku])
                ))
        
        fig_inv.update_layout(
            xaxis_title="Day",
            yaxis_title="Inventory (units)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_inv, use_container_width=True)
        
    elif 'warehouses' in history:
        tabs = st.tabs(["🏢 Warehouse East", "🏢 Warehouse West", "🏢 Warehouse Central"])
        colors = {'PROD_A': '#3498db', 'PROD_B': '#2ecc71', 'PROD_C': '#9b59b6'}
        
        for wh_idx, tab in enumerate(tabs):
            with tab:
                fig_wh = go.Figure()
                
                for sku in ['PROD_A', 'PROD_B', 'PROD_C']:
                    wh_key = f'WH_{wh_idx}'
                    if wh_key in history['warehouses']:
                        fig_wh.add_trace(go.Scatter(
                            x=history['days'],
                            y=history['warehouses'][wh_key][sku],
                            mode='lines',
                            name=sku,
                            line=dict(width=2, color=colors[sku])
                        ))
                
                fig_wh.update_layout(
                    xaxis_title="Day",
                    yaxis_title="Inventory (units)",
                    height=350,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_wh, use_container_width=True)
    
    st.subheader("📊 Daily Cost Breakdown")
    
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Bar(
        x=history['days'],
        y=history['costs'],
        name='Daily Cost',
        marker_color='#3498db',
        opacity=0.7
    ))
    
    if len(history['costs']) > 7:
        moving_avg = pd.Series(history['costs']).rolling(window=7).mean()
        fig_daily.add_trace(go.Scatter(
            x=history['days'],
            y=moving_avg,
            mode='lines',
            name='7-Day Moving Avg',
            line=dict(color='#e74c3c', width=3)
        ))
    
    fig_daily.update_layout(
        xaxis_title="Day",
        yaxis_title="Daily Cost ($)",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_daily, use_container_width=True)


def show_simulation_page(results):
    """Live simulation page"""
    st.header("🎮 Live Agent Simulation")
    
    st.info("💡 Select an agent and watch it make inventory decisions in real-time!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        agent_choice = st.selectbox(
            "Select Agent:",
            [
                "Random Policy",
                "Reorder Point Policy",
                "DQN Agent (Single Warehouse)",
                "Independent Multi-Agent (3 Warehouses)",
                "Coordinated Multi-Agent (3 Warehouses)"
            ]
        )
    
    with col2:
        episode_length = st.slider("Days:", 30, 180, 60, 30)
    
    if st.button("▶️ Run Simulation", type="primary", use_container_width=True):
        
        with st.spinner(f"🔄 Running {agent_choice} for {episode_length} days..."):
            
            try:
                if "Random" in agent_choice:
                    env = SupplyChainEnv(num_products=3, episode_length=episode_length, random_seed=42)
                    policies = create_baseline_policies(env)
                    history = run_baseline_simulation(env, policies['random'], episode_length)
                    
                elif "Reorder Point" in agent_choice:
                    env = SupplyChainEnv(num_products=3, episode_length=episode_length, random_seed=42)
                    policies = create_baseline_policies(env)
                    history = run_baseline_simulation(env, policies['reorder_point'], episode_length)
                    
                elif "DQN Agent" in agent_choice:
                    if os.path.exists('models/dqn_optimized.zip'):
                        env = SupplyChainEnv(num_products=3, episode_length=episode_length, random_seed=42)
                        env = FlattenMultiDiscreteWrapper(env)
                        model = DQN.load('models/dqn_optimized.zip', env=env)
                        history = run_dqn_simulation(env, model, episode_length)
                    else:
                        st.error("❌ DQN model not found!")
                        return
                
                elif "Independent" in agent_choice:
                    if os.path.exists('models/independent_multi_agent.zip'):
                        env = MultiWarehouseEnv(num_products=3, num_warehouses=3,
                                               episode_length=episode_length, enable_transfers=False, random_seed=42)
                        
                        agent = IndependentMultiAgent(env)
                        temp_env = SupplyChainEnv(num_products=3, episode_length=180)
                        temp_env = FlattenMultiDiscreteWrapper(temp_env)
                        agent.agent = DQN.load('models/independent_multi_agent.zip', env=temp_env)
                        
                        history = run_multiagent_simulation(env, agent, episode_length)
                    else:
                        st.error("❌ Independent model not found!")
                        return
                
                elif "Coordinated" in agent_choice:
                    if os.path.exists('models/coordinated_multi_agent.zip'):
                        env = MultiWarehouseEnv(num_products=3, num_warehouses=3,
                                               episode_length=episode_length, enable_transfers=True, random_seed=42)
                        
                        agent = CoordinatedMultiAgent(env)
                        wrapped = FlattenMultiDiscreteWrapper(env)
                        agent.agent = DQN.load('models/coordinated_multi_agent.zip', env=wrapped)
                        
                        history = run_multiagent_simulation(env, agent, episode_length)
                    else:
                        st.error("❌ Coordinated model not found!")
                        return
                
                display_simulation_results(history, agent_choice, episode_length)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                with st.expander("🔍 Details"):
                    import traceback
                    st.code(traceback.format_exc())
    
    else:
        st.markdown("---")
        st.markdown("### 👆 Click **Run Simulation** above!")
        st.markdown("""
        **You'll see:**
        - 📊 Real-time cost accumulation
        - 📦 Inventory levels per product
        - 💰 Daily cost breakdown
        - 🔄 Transfer activity (multi-agent only)
        """)


def show_learning_curves_page(results):
    """Learning curves page"""
    st.header("📈 Learning Curves")
    
    if Path('data/learning_curves.png').exists():
        st.image('data/learning_curves.png', use_container_width=True)
    
    st.markdown("---")
    st.subheader("📊 Training Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **DQN Training:**
        - Episodes: ~1,110
        - Timesteps: 200,000
        - Final: $2.27M
        - Improvement: 57%
        - Time: ~40 min
        """)
    
    with col2:
        st.markdown("""
        **Multi-Agent:**
        - Episodes: ~833
        - Timesteps: 150,000
        - Final: $12.91M
        - Transfers: 326/ep
        - Time: ~50 min
        """)
    
    if Path('data/complete_comparison.png').exists():
        st.markdown("---")
        st.image('data/complete_comparison.png', use_container_width=True)


def show_scenario_page(results):
    """Scenario page"""
    st.header("🧪 Scenario Testing")
    
    try:
        df_scenarios = pd.read_csv('data/scenario_results.csv')
        
        selected_scenario = st.selectbox(
            "Select Scenario:",
            df_scenarios['scenario'].unique(),
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        scenario_data = df_scenarios[df_scenarios['scenario'] == selected_scenario]
        
        fig = px.bar(
            scenario_data,
            x='policy',
            y='mean_cost',
            error_y='std_cost',
            title=f'{selected_scenario.replace("_", " ").title()}',
            color='mean_cost',
            color_continuous_scale='RdYlGn_r',
            text='mean_cost'
        )
        
        fig.update_traces(texttemplate='$%{text:.2s}', textposition='outside')
        fig.update_layout(height=500)
        fig.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig, use_container_width=True)
        
        best_idx = scenario_data['mean_cost'].idxmin()
        winner = scenario_data.loc[best_idx]
        st.success(f"🏆 Winner: {winner['policy'].title()} - ${winner['mean_cost']:,.0f}")
        
    except FileNotFoundError:
        st.error("Scenario results not found")
    
    if Path('data/scenario_comparison_bars.png').exists():
        st.markdown("---")
        st.image('data/scenario_comparison_bars.png', use_container_width=True)


def show_ablation_page(results):
    """Ablation page"""
    st.header("🔍 Ablation Study")
    
    if results.get('ablation'):
        ablation = results['ablation']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "WITH Transfers",
                f"${ablation['with_transfers']['mean_cost']/1e6:.2f}M",
                f"{int(ablation['with_transfers']['mean_transfers'])} transfers"
            )
        
        with col2:
            st.metric(
                "WITHOUT Transfers",
                f"${ablation['without_transfers']['mean_cost']/1e6:.2f}M",
                "0 transfers"
            )
        
        with col3:
            impact = ablation['transfer_impact']['percent_difference']
            st.metric(
                "Impact",
                f"{abs(impact):.1f}%",
                "Benefit" if impact < 0 else "Overhead",
                delta_color="normal" if impact < 0 else "inverse"
            )
        
        st.markdown("---")
        
        if Path('data/ablation_visualization.png').exists():
            st.image('data/ablation_visualization.png', use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <h4>💡 Key Finding</h4>
        <p><b>Transfers save 10.3%</b> ($1.46M), proving the mechanism works. 
        But coordinated agent is still 130% worse than independent - it learned 
        to over-rely on transfers (326/episode) instead of better ordering.</p>
        <p><b>Lesson:</b> Coordination tools work, but agents must learn to use them wisely.</p>
        </div>
        """, unsafe_allow_html=True)


def show_technical_page(results):
    """Technical page"""
    st.header("📋 Technical Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Environment:**
        - Products: 3 SKUs
        - Warehouses: 1 or 3
        - Episode: 180 days
        - State: 13/57 dims
        - Actions: 64/262K
        """)
    
    with col2:
        st.markdown("""
        **DQN Config:**
        - Network: [512, 512, 256]
        - LR: 0.0003
        - Buffer: 100K
        - Batch: 128
        - γ: 0.99
        """)
    
    st.markdown("---")
    
    st.subheader("📊 Results Summary")
    
    try:
        df = pd.read_csv('data/statistical_summary.csv')
        st.dataframe(df, use_container_width=True, hide_index=True)
    except:
        pass
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if Path('data/results_summary.csv').exists():
            with open('data/results_summary.csv') as f:
                st.download_button(
                    "📊 Results CSV",
                    f.read(),
                    "results.csv",
                    use_container_width=True
                )
    
    with col2:
        if Path('data/statistical_analysis.json').exists():
            with open('data/statistical_analysis.json') as f:
                st.download_button(
                    "📈 Statistics JSON",
                    f.read(),
                    "statistics.json",
                    use_container_width=True
                )
    
    with col3:
        if Path('data/multi_agent_results.json').exists():
            with open('data/multi_agent_results.json') as f:
                st.download_button(
                    "🤝 Multi-Agent JSON",
                    f.read(),
                    "multi_agent.json",
                    use_container_width=True
                )


if __name__ == "__main__":
    main()
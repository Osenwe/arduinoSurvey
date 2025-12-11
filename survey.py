import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from scipy import stats
from statsmodels.stats.power import TTestPower
import io

# Page configuration
st.set_page_config(
    page_title="Arduino Survey Analysis",
    page_icon="🤖",
    layout="wide"
)

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def gradient_color(values, base_color='blue'):
    """Generate gradient colors where higher values get darker colors"""
    color_maps = {
        'blue': plt.cm.Blues,
        'green': plt.cm.Greens,
        'purple': plt.cm.Purples,
        'orange': plt.cm.Oranges,
        'red': plt.cm.Reds
    }
    
    cmap = color_maps.get(base_color, plt.cm.Blues)
    max_val = max(values) if max(values) > 0 else 1
    min_val = min(values)
    
    colors = []
    for val in values:
        if max_val == min_val:
            intensity = 0.5
        else:
            intensity = 0.25 + 0.6 * (val - min_val) / (max_val - min_val)
        colors.append(cmap(intensity))
    
    return colors


def load_and_prepare_data(df):
    """Load and prepare survey data"""
    df.columns = ['timestamp', 'q1_familiarity', 'q2_confidence', 'q3_enjoyment', 'q4_helpful', 'q5_challenges']
    return df


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def create_overview_charts(df):
    """Create the 2x2 overview chart with Q1, Q2, Q3, and summary stats"""
    fig = plt.figure(figsize=(14, 12))
    
    n = len(df)
    
    # Q1: Prior Familiarity
    ax1 = fig.add_subplot(2, 2, 1)
    familiarity_counts = df['q1_familiarity'].value_counts().sort_index()
    familiarity_labels = ['Not familiar', 'A little familiar', 'Somewhat familiar', 'Very familiar']
    familiarity_values = [familiarity_counts.get(i, 0) for i in range(1, 5)]
    colors_familiarity = gradient_color(familiarity_values, 'blue')
    
    bars1 = ax1.bar(familiarity_labels, familiarity_values, color=colors_familiarity, 
                    edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Number of Students', fontsize=11)
    ax1.set_title('Q1: Prior Familiarity with Arduino\n(Before Cycle 2)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(familiarity_values) + 1)
    
    for bar, val in zip(bars1, familiarity_values):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(val),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)
    
    # Q2: Confidence Level
    ax2 = fig.add_subplot(2, 2, 2)
    confidence_counts = df['q2_confidence'].value_counts().sort_index()
    confidence_labels = ['Not confident', 'A little confident', 'Confident', 'Very confident']
    confidence_values = [confidence_counts.get(i, 0) for i in range(1, 5)]
    colors_confidence = gradient_color(confidence_values, 'green')
    
    bars2 = ax2.bar(confidence_labels, confidence_values, color=colors_confidence, 
                    edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Number of Students', fontsize=11)
    ax2.set_title('Q2: Confidence Using Arduino Components\n(After Activity)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(confidence_values) + 1)
    
    for bar, val in zip(bars2, confidence_values):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(val),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    
    # Q3: Enjoyment Level
    ax3 = fig.add_subplot(2, 2, 3)
    enjoyment_counts = df['q3_enjoyment'].value_counts().sort_index()
    enjoyment_labels = ['1\n(Not enjoyable)', '2', '3', '4', '5\n(Very enjoyable)']
    enjoyment_values = [enjoyment_counts.get(i, 0) for i in range(1, 6)]
    colors_enjoyment = gradient_color(enjoyment_values, 'purple')
    
    bars3 = ax3.bar(enjoyment_labels, enjoyment_values, color=colors_enjoyment, 
                    edgecolor='black', linewidth=1.2)
    ax3.set_ylabel('Number of Students', fontsize=11)
    ax3.set_xlabel('Enjoyment Rating', fontsize=11)
    ax3.set_title('Q3: Enjoyment of Hands-on Arduino Activity', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, max(enjoyment_values) + 1)
    
    for bar, val in zip(bars3, enjoyment_values):
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(val),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Summary Statistics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    mean_familiarity = df['q1_familiarity'].mean()
    mean_confidence = df['q2_confidence'].mean()
    mean_enjoyment = df['q3_enjoyment'].mean()
    
    low_familiarity = df[df['q1_familiarity'] <= 2]
    confidence_gain = (low_familiarity['q2_confidence'] >= 3).sum()
    confidence_gain_pct = (confidence_gain / len(low_familiarity)) * 100 if len(low_familiarity) > 0 else 0
    
    high_enjoyment = (df['q3_enjoyment'] >= 4).sum()
    high_enjoyment_pct = (high_enjoyment / n) * 100
    
    confident_students = (df['q2_confidence'] >= 3).sum()
    confident_pct = (confident_students / n) * 100
    
    stats_text = f"""
SURVEY SUMMARY STATISTICS
{'='*40}

Sample Size: n = {n} students

PRIOR FAMILIARITY (Q1)
• Mean: {mean_familiarity:.2f} / 4.00
• Not familiar (1): {familiarity_values[0]} students ({familiarity_values[0]/n*100:.1f}%)
• A little familiar (2): {familiarity_values[1]} students ({familiarity_values[1]/n*100:.1f}%)
• Somewhat familiar (3): {familiarity_values[2]} students ({familiarity_values[2]/n*100:.1f}%)
• Very familiar (4): {familiarity_values[3]} students ({familiarity_values[3]/n*100:.1f}%)

CONFIDENCE AFTER ACTIVITY (Q2)
• Mean: {mean_confidence:.2f} / 4.00
• Students at "Confident" or higher: {confident_students} ({confident_pct:.1f}%)
• Students with low prior familiarity who
  gained confidence (≥3): {confidence_gain}/{len(low_familiarity)} ({confidence_gain_pct:.1f}%)

ENJOYMENT (Q3)
• Mean: {mean_enjoyment:.2f} / 5.00
• Students rating 4 or 5: {high_enjoyment} ({high_enjoyment_pct:.1f}%)
"""
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    
    plt.tight_layout()
    return fig


def create_enjoyment_pie_chart(df):
    """Create pie chart for enjoyment distribution"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    enjoyment_counts = df['q3_enjoyment'].value_counts().sort_index()
    enjoyment_labels_pie = [f'Rating {i}' for i in range(1, 6)]
    enjoyment_values_pie = [enjoyment_counts.get(i, 0) for i in range(1, 6)]
    
    colors_enjoyment = gradient_color(enjoyment_values_pie, 'purple')
    
    labels_filtered = []
    values_filtered = []
    colors_filtered = []
    for label, val, color in zip(enjoyment_labels_pie, enjoyment_values_pie, colors_enjoyment):
        if val > 0:
            labels_filtered.append(f'{label}\n({val} students)')
            values_filtered.append(val)
            colors_filtered.append(color)
    
    wedges, texts, autotexts = ax.pie(values_filtered, labels=labels_filtered, 
                                       colors=colors_filtered, autopct='%1.1f%%', 
                                       startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    
    ax.set_title(f'Q3: Distribution of Enjoyment Ratings\n(n={len(df)} students)', 
                 fontsize=13, fontweight='bold', pad=20)
    
    return fig


def create_wordclouds(df):
    """Create word clouds for Q4 and Q5"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    stopwords = set(STOPWORDS)
    stopwords.update(['helped', 'understand', 'engineering', 'using', 'activity', 
                      'arduino', 'component', 'made', 'really', 'lot', 'also'])
    
    # Q4 Word Cloud
    q4_text = ' '.join(df['q4_helpful'].dropna().astype(str))
    if q4_text.strip():
        wc_q4 = WordCloud(width=800, height=400, background_color='white', 
                         colormap='Greens', stopwords=stopwords, 
                         max_words=50).generate(q4_text)
        ax1.imshow(wc_q4, interpolation='bilinear')
        ax1.set_title('Q4: What Helped Most in Understanding Engineering', 
                     fontsize=13, fontweight='bold', pad=15)
    ax1.axis('off')
    
    # Q5 Word Cloud
    stopwords_q5 = stopwords.copy()
    stopwords_q5.update(['challenge', 'challenging', 'difficult', 'hard', 'faced', 'facing'])
    
    q5_text = ' '.join(df['q5_challenges'].dropna().astype(str))
    if q5_text.strip():
        wc_q5 = WordCloud(width=800, height=400, background_color='white', 
                         colormap='Oranges', stopwords=stopwords_q5, 
                         max_words=50).generate(q5_text)
        ax2.imshow(wc_q5, interpolation='bilinear')
        ax2.set_title('Q5: Challenges Faced During Activity', 
                     fontsize=13, fontweight='bold', pad=15)
    ax2.axis('off')
    
    plt.tight_layout()
    return fig


def create_thematic_analysis(df):
    """Create thematic analysis with frequency and co-occurrence patterns"""
    n = len(df)
    
    # Define themes
    themes_q4 = ['Hands-on', 'Visual', 'Teamwork', 'Instructions', 'Real-world']
    themes_q5 = ['Coding', 'Wiring', 'Debugging', 'Time', 'Components']
    
    # Q4 Theme detection
    q4_freq = {theme: 0 for theme in themes_q4}
    theme_matrix_q4 = []
    
    for _, row in df.iterrows():
        response_lower = str(row['q4_helpful']).lower()
        row_themes = [
            1 if any(word in response_lower for word in ['hand', 'touch', 'physical', 'practice']) else 0,
            1 if any(word in response_lower for word in ['see', 'visual', 'watch', 'observe', 'diagram']) else 0,
            1 if any(word in response_lower for word in ['team', 'group', 'partner', 'together', 'collaborate']) else 0,
            1 if any(word in response_lower for word in ['instruction', 'guide', 'step', 'direction', 'explain']) else 0,
            1 if any(word in response_lower for word in ['real', 'application', 'practical', 'use', 'apply']) else 0
        ]
        theme_matrix_q4.append(row_themes)
        for i, theme in enumerate(themes_q4):
            q4_freq[theme] += row_themes[i]
    
    df_themes_q4 = pd.DataFrame(theme_matrix_q4, columns=themes_q4)
    cooccur_q4 = df_themes_q4.T.dot(df_themes_q4)
    
    # Q5 Theme detection
    q5_freq = {theme: 0 for theme in themes_q5}
    theme_matrix_q5 = []
    
    for _, row in df.iterrows():
        response_lower = str(row['q5_challenges']).lower()
        row_themes = [
            1 if any(word in response_lower for word in ['code', 'coding', 'program', 'syntax']) else 0,
            1 if any(word in response_lower for word in ['wire', 'wiring', 'connect', 'cable']) else 0,
            1 if any(word in response_lower for word in ['debug', 'error', 'troubleshoot', 'fix', 'problem']) else 0,
            1 if any(word in response_lower for word in ['time', 'rush', 'quick', 'fast']) else 0,
            1 if any(word in response_lower for word in ['component', 'part', 'piece', 'fit', 'fall']) else 0
        ]
        theme_matrix_q5.append(row_themes)
        for i, theme in enumerate(themes_q5):
            q5_freq[theme] += row_themes[i]
    
    df_themes_q5 = pd.DataFrame(theme_matrix_q5, columns=themes_q5)
    cooccur_q5 = df_themes_q5.T.dot(df_themes_q5)
    
    # Create visualization
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Q4 Bar Chart
    ax1 = fig.add_subplot(gs[0, 0])
    sorted_q4 = sorted(q4_freq.items(), key=lambda x: x[1])
    themes_q4_sorted = [item[0] for item in sorted_q4]
    counts_q4_sorted = [item[1] for item in sorted_q4]
    colors_q4 = gradient_color(counts_q4_sorted, 'green')
    
    bars_q4 = ax1.barh(themes_q4_sorted, counts_q4_sorted, color=colors_q4,
                        edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Number of Mentions', fontsize=11)
    ax1.set_title('Q4: Theme Frequency - What Helped Most', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, max(counts_q4_sorted) + 1)
    
    for bar, val in zip(bars_q4, counts_q4_sorted):
        if val > 0:
            ax1.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2, str(val),
                    va='center', fontsize=11, fontweight='bold')
    
    # Q4 Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(cooccur_q4, annot=True, fmt='d', cmap='Greens',
                xticklabels=themes_q4, yticklabels=themes_q4, ax=ax2,
                cbar_kws={'label': 'Co-occurrence'},
                vmin=0, vmax=max(cooccur_q4.max().max(), cooccur_q5.max().max()),
                linewidths=0.5, linecolor='gray', square=True)
    ax2.set_title('Q4: Theme Co-occurrence\nWhat Helped Understand Engineering',
                  fontweight='bold', fontsize=12)
    
    # Q5 Bar Chart
    ax3 = fig.add_subplot(gs[1, 0])
    sorted_q5 = sorted(q5_freq.items(), key=lambda x: x[1])
    themes_q5_sorted = [item[0] for item in sorted_q5]
    counts_q5_sorted = [item[1] for item in sorted_q5]
    colors_q5 = gradient_color(counts_q5_sorted, 'orange')
    
    bars_q5 = ax3.barh(themes_q5_sorted, counts_q5_sorted, color=colors_q5,
                        edgecolor='black', linewidth=1.2)
    ax3.set_xlabel('Number of Mentions', fontsize=11)
    ax3.set_title('Q5: Theme Frequency - Challenges Faced', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, max(counts_q5_sorted) + 1)
    
    for bar, val in zip(bars_q5, counts_q5_sorted):
        if val > 0:
            ax3.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2, str(val),
                    va='center', fontsize=11, fontweight='bold')
    
    # Q5 Heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    sns.heatmap(cooccur_q5, annot=True, fmt='d', cmap='Oranges',
                xticklabels=themes_q5, yticklabels=themes_q5, ax=ax4,
                cbar_kws={'label': 'Co-occurrence'},
                vmin=0, vmax=max(cooccur_q4.max().max(), cooccur_q5.max().max()),
                linewidths=0.5, linecolor='gray', square=True)
    ax4.set_title('Q5: Theme Co-occurrence\nChallenges Faced During Activity',
                  fontweight='bold', fontsize=12)
    
    plt.suptitle(f'Thematic Analysis: Frequency and Co-occurrence Patterns (n={n})',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Generate insights
    all_themes_q4 = df_themes_q4.sum(axis=1)
    multi_theme_q4 = (all_themes_q4 >= 2).sum()
    multi_theme_q4_pct = multi_theme_q4 / n * 100
    
    all_themes_q5 = df_themes_q5.sum(axis=1)
    multi_theme_q5 = (all_themes_q5 >= 2).sum()
    multi_theme_q5_pct = multi_theme_q5 / n * 100
    
    insights = {
        'q4_freq': q4_freq,
        'q5_freq': q5_freq,
        'multi_theme_q4': multi_theme_q4,
        'multi_theme_q4_pct': multi_theme_q4_pct,
        'multi_theme_q5': multi_theme_q5,
        'multi_theme_q5_pct': multi_theme_q5_pct,
        'cooccur_q4': cooccur_q4,
        'cooccur_q5': cooccur_q5,
        'themes_q4': themes_q4,
        'themes_q5': themes_q5
    }
    
    return fig, insights


# ============================================================
# MAIN APP
# ============================================================

def main():
    st.title("🤖 Arduino Survey Analysis Dashboard")
    st.markdown("### Analyzing Student Responses to Arduino Engineering Activities")
    
    # Sidebar
    st.sidebar.header("📁 Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        df = load_and_prepare_data(df)
        
        st.sidebar.success(f"✅ Loaded {len(df)} responses")
        
        # Display raw data option
        if st.sidebar.checkbox("Show Raw Data"):
            st.subheader("📊 Raw Survey Data")
            st.dataframe(df)
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Overview", 
            "🎯 Enjoyment Analysis", 
            "☁️ Word Clouds", 
            "🔍 Thematic Analysis"
        ])
        
        with tab1:
            st.subheader("Survey Overview")
            fig = create_overview_charts(df)
            st.pyplot(fig)
            plt.close()
            
            # Download button
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            st.download_button(
                label="📥 Download Overview Chart",
                data=buf,
                file_name="survey_overview.png",
                mime="image/png"
            )
        
        with tab2:
            st.subheader("Enjoyment Distribution")
            fig = create_enjoyment_pie_chart(df)
            st.pyplot(fig)
            plt.close()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            st.download_button(
                label="📥 Download Enjoyment Chart",
                data=buf,
                file_name="enjoyment_pie_chart.png",
                mime="image/png"
            )
        
        with tab3:
            st.subheader("Word Cloud Analysis")
            fig = create_wordclouds(df)
            st.pyplot(fig)
            plt.close()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            st.download_button(
                label="📥 Download Word Clouds",
                data=buf,
                file_name="wordclouds.png",
                mime="image/png"
            )
        
        with tab4:
            st.subheader("Thematic Analysis")
            fig, insights = create_thematic_analysis(df)
            st.pyplot(fig)
            plt.close()
            
            # Display insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 💡 Q4 Insights: What Helped")
                st.metric("Students mentioning multiple themes", 
                         f"{insights['multi_theme_q4']} ({insights['multi_theme_q4_pct']:.1f}%)")
                
                st.markdown("**Top Themes:**")
                for theme, count in sorted(insights['q4_freq'].items(), key=lambda x: x[1], reverse=True)[:3]:
                    st.write(f"• {theme}: {count} mentions")
            
            with col2:
                st.markdown("#### ⚠️ Q5 Insights: Challenges")
                st.metric("Students facing multiple challenges", 
                         f"{insights['multi_theme_q5']} ({insights['multi_theme_q5_pct']:.1f}%)")
                
                st.markdown("**Top Challenges:**")
                for theme, count in sorted(insights['q5_freq'].items(), key=lambda x: x[1], reverse=True)[:3]:
                    st.write(f"• {theme}: {count} mentions")
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            st.download_button(
                label="📥 Download Thematic Analysis",
                data=buf,
                file_name="thematic_analysis.png",
                mime="image/png"
            )
    
    else:
        st.info("👆 Please upload a CSV file to begin analysis")
        st.markdown("""
        ### Expected CSV Format
        The CSV should contain the following columns:
        1. **Timestamp** - Survey submission time
        2. **Q1: Familiarity** - Prior familiarity with Arduino (1-4 scale)
        3. **Q2: Confidence** - Confidence level after activity (1-4 scale)
        4. **Q3: Enjoyment** - Enjoyment rating (1-5 scale)
        5. **Q4: Helpful** - Open-ended: What helped understand engineering
        6. **Q5: Challenges** - Open-ended: Challenges faced during activity
        """)


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100


# DATA LOADING & CLEANING
# ============================================================================

# Import data
df = pd.read_csv('data/fcc-forum-pageviews.csv', parse_dates=['date'], index_col='date')

print("="*60)
print("📊 TIME SERIES VISUALIZER - freeCodeCamp Forum Page Views")
print("="*60)
print(f"\n📅 Original dataset:")
print(f"   Rows: {len(df):,}")
print(f"   Date range: {df.index.min().date()} to {df.index.max().date()}")
print(f"   Page views: {df['value'].min():,} to {df['value'].max():,}")

# Clean data (remove top/bottom 2.5% outliers)
lower_bound = df['value'].quantile(0.025)
upper_bound = df['value'].quantile(0.975)
df = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]

print(f"\n🧹 After cleaning (removed top/bottom 2.5%):")
print(f"   Rows: {len(df):,}")
print(f"   Page views: {df['value'].min():,} to {df['value'].max():,}")


# FUNCTION 1: LINE PLOT
# ============================================================================

def draw_line_plot():
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df.index, df['value'], color='#d62728', linewidth=1)
    
    ax.set_title('Daily freeCodeCamp Forum Page Views 5/2016-12/2019',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Page Views', fontsize=12, fontweight='bold')
    
    # Format y-axis with commas
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    fig.savefig('line_plot.png', bbox_inches='tight')
    return fig


# FUNCTION 2: BAR PLOT
# ============================================================================

def draw_bar_plot():
    
    # Copy and prepare data
    df_bar = df.copy()
    df_bar['year'] = df_bar.index.year
    df_bar['month'] = df_bar.index.month
    
    # Group by year and month, calculate mean
    df_bar = df_bar.groupby(['year', 'month'])['value'].mean()
    df_bar = df_bar.unstack()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    df_bar.plot(kind='bar', ax=ax, width=0.75, colormap='tab10', legend=True)
    
    ax.set_xlabel('Years', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Page Views', fontsize=12, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Legend
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.legend(title='Months', labels=month_names, 
              loc='upper left', fontsize=9, title_fontsize=10)
    
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    
    fig.savefig('bar_plot.png', bbox_inches='tight')
    return fig


# FUNCTION 3: BOX PLOTS
# ============================================================================

def draw_box_plot():
    
    # Prepare data
    df_box = df.copy()
    df_box.reset_index(inplace=True)
    df_box['year'] = [d.year for d in df_box['date']]
    df_box['month'] = [d.strftime('%b') for d in df_box['date']]
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Year-wise box plot
    sns.boxplot(x='year', y='value', data=df_box, ax=axes[0], palette='Set2', linewidth=1.5)
    
    axes[0].set_title('Year-wise Box Plot (Trend)', fontsize=14, fontweight='bold', pad=12)

    axes[0].set_xlabel('Year', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Page Views', fontsize=12, fontweight='bold')

    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    axes[0].grid(True, alpha=0.2, axis='y')
    
    # Month-wise box plot
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    sns.boxplot(x='month', y='value', data=df_box, ax=axes[1], order=month_order, palette='Set3', linewidth=1.5)
    
    axes[1].set_title('Month-wise Box Plot (Seasonality)', 
                     fontsize=14, fontweight='bold', pad=12)
    axes[1].set_xlabel('Month', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Page Views', fontsize=12, fontweight='bold')
    axes[1].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    axes[1].grid(True, alpha=0.2, axis='y')
    
    plt.tight_layout()
    fig.savefig('box_plot.png', bbox_inches='tight')
    return fig


# GROWTH RATE ANALYSIS
# ============================================================================

def analyze_growth():

    yearly_avg = df.groupby(df.index.year)['value'].mean()
    
    print("📈 YEAR-OVER-YEAR GROWTH ANALYSIS")
    print("="*60)
    
    for i, year in enumerate(yearly_avg.index):
        avg = yearly_avg[year]
        print(f"\n{year}: {avg:>10,.0f} avg daily views", end="")
        
        if i > 0:
            prev_avg = yearly_avg[yearly_avg.index[i-1]]
            growth = ((avg - prev_avg) / prev_avg) * 100
            arrow = "↗" if growth > 0 else "↘"
            print(f"  {arrow} {growth:+6.1f}% from {year-1}")
        else:
            print()
    
    # Total growth
    total_growth = ((yearly_avg.iloc[-1] - yearly_avg.iloc[0]) / yearly_avg.iloc[0]) * 100
    print(f"Total growth (2016→2019): {total_growth:+.1f}%")


# BONUS: PEAK TRAFFIC DAYS
# ============================================================================

def identify_peaks():
    top_10 = df.nlargest(10, 'value')
    
    print("🔥 TOP 10 HIGHEST TRAFFIC DAYS")
    print("="*60)
    
    for i, (date, row) in enumerate(top_10.iterrows(), 1):
        views = row['value']
        weekday = date.strftime('%A')
        month_day = date.strftime('%B %d, %Y')
        print(f"{i:2}. {month_day:>20} ({weekday:>9}): {views:>9,} views")
    
    print("="*60)


# BONUS: DAY-OF-WEEK PATTERN
# ============================================================================

def analyze_weekday_pattern():
    df_week = df.copy()
    df_week['weekday'] = df_week.index.day_name()
    
    # Calculate average by weekday
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_avg = df_week.groupby('weekday')['value'].mean().reindex(weekday_order)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    colors = ['#1f77b4' if day not in ['Saturday', 'Sunday'] else '#ff7f0e' for day in weekday_order]
    
    bars = ax.bar(range(7), weekday_avg.values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(range(7))
    ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], fontsize=11, fontweight='bold')

    ax.set_ylabel('Average Page Views', fontsize=12, fontweight='bold')
    ax.set_title('Average Page Views by Day of Week', fontsize=14, fontweight='bold', pad=15)
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1f77b4', label='Weekday'),
                      Patch(facecolor='#ff7f0e', label='Weekend')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    fig.savefig('weekday_pattern.png', bbox_inches='tight')
    
    print("\n" + "="*60)
    print("📅 WEEKDAY TRAFFIC PATTERN")
    print("="*70)
    for day in weekday_order:
        avg = weekday_avg[day]
        day_type = "Weekend" if day in ['Saturday', 'Sunday'] else "Weekday"
        print(f"{day:>9} ({day_type:>7}): {avg:>8,.0f} avg views")
    
    weekend_avg = weekday_avg[['Saturday', 'Sunday']].mean()
    weekday_avg_val = weekday_avg[['Monday', 'Tuesday', 'Wednesday', 
                                     'Thursday', 'Friday']].mean()
    difference = ((weekend_avg - weekday_avg_val) / weekday_avg_val) * 100
    
    print(f"\n{'─'*70}")
    print(f"Weekday average: {weekday_avg_val:>8,.0f}")
    print(f"Weekend average: {weekend_avg:>8,.0f}")
    print(f"Weekend vs Weekday: {difference:+.1f}%")
    print("="*70)
    
    return fig


# MONTHLY TREND HEATMAP
# ============================================================================

def draw_monthly_heatmap():
    df_heat = df.copy()
    df_heat['year'] = df_heat.index.year
    df_heat['month'] = df_heat.index.month
    
    # Pivot to create matrix
    heat_data = df_heat.groupby(['year', 'month'])['value'].mean().unstack()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    sns.heatmap(heat_data, annot=True, fmt='.0f', cmap='YlOrRd',
                linewidths=1, linecolor='white', cbar_kws={'label': 'Avg Views'},
                ax=ax)
    
    ax.set_title('Average Daily Page Views - Monthly Heatmap', fontsize=14, fontweight='bold', pad=15)

    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Year', fontsize=12, fontweight='bold')
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticklabels(month_names, rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    fig.savefig('monthly_heatmap.png', bbox_inches='tight')
    
    return fig


# SUMMARY STATISTICS
# ============================================================================

def print_summary():
    
    print("📊 SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\n📈 Overall Stats:")
    print(f"   Mean:   {df['value'].mean():>10,.0f} views/day")
    print(f"   Median: {df['value'].median():>10,.0f} views/day")
    print(f"   Std:    {df['value'].std():>10,.0f}")
    print(f"   Min:    {df['value'].min():>10,} views")
    print(f"   Max:    {df['value'].max():>10,} views")
    
    print(f"\n📅 Coverage:")
    print(f"   Total days analyzed: {len(df):,}")
    print(f"   Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"   Years covered: {df.index.year.nunique()}")
    
    # Busiest and quietest months
    monthly_avg = df.groupby(df.index.month)['value'].mean()
    busiest_month = monthly_avg.idxmax()
    quietest_month = monthly_avg.idxmin()
    
    month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    
    print(f"\n📆 Seasonal Insights:")
    print(f"   Busiest month:  {month_names[busiest_month]} " f"({monthly_avg[busiest_month]:,.0f} avg)")
    print(f"   Quietest month: {month_names[quietest_month]} " f"({monthly_avg[quietest_month]:,.0f} avg)")
    
    print("="*60)


# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    
    # Required freeCodeCamp plots
    print("\n🎨 Generating required visualizations...")
    draw_line_plot()
    print("   ✓ Line plot saved → line_plot.png")
    
    draw_bar_plot()
    print("   ✓ Bar plot saved → bar_plot.png")
    
    draw_box_plot()
    print("   ✓ Box plot saved → box_plot.png")
    
    # Bonus analyses
    print("\n🔍 Running bonus analyses...")
    analyze_growth()
    identify_peaks()
    analyze_weekday_pattern()
    print("   ✓ Weekday pattern saved → weekday_pattern.png")
    
    draw_monthly_heatmap()
    print("   ✓ Monthly heatmap saved → monthly_heatmap.png")
    
    print_summary()
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  📈 line_plot.png         - Daily trend")
    print("  📊 bar_plot.png          - Monthly averages")
    print("  📦 box_plot.png          - Distributions")
    print("  📅 weekday_pattern.png   - Day-of-week analysis")
    print("  🔥 monthly_heatmap.png    - Year×Month heatmap")
    print("\n" + "="*60)


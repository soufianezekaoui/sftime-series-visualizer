import time_series_visualizer
from unittest import main


# Test my functions
time_series_visualizer.draw_line_plot()
time_series_visualizer.draw_bar_plot()
time_series_visualizer.draw_box_plot()
time_series_visualizer.draw_monthly_heatmap()
time_series_visualizer.analyze_weekday_pattern()
time_series_visualizer.identify_peaks()
time_series_visualizer.print_summary()


# Run unit tests automatically
main(module='test_module', exit=False)

import matplotlib.pyplot as plt

def visualize_percentage(labels, percentages):
    # Define custom colors for the pie chart
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Create a pie chart
    plt.figure(figsize=(8, 8))
    explode = [0.05] * len(labels)  # Explode each slice for a "popping out" effect
    plt.pie(percentages, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140, explode=explode,
            shadow=True, wedgeprops=dict(width=0.4, edgecolor='w'))

    # Add a title and a legend
    plt.title("Percentage of Data")
    plt.legend(labels, loc="best")

    # Display the chart
    plt.axis("equal")
    plt.show()

if __name__ == "__main__":
    # Sample data - replace with your actual data
    labels = ["Label A", "Label B", "Label C", "Label D"]
    percentages = [25, 30, 20, 25]

    visualize_percentage(labels, percentages)

# import plotly.graph_objects as go
# import numpy as np

# def create_animated_pie_chart(labels, percentages):
#     # Create a gradient color scheme for the pie slices
#     colors = [f"hsl({360 * i / len(labels)}, 50%, 60%)" for i in range(len(labels))]

#     # Create the pie chart
#     fig = go.Figure()

#     fig.add_trace(go.Pie(
#         labels=labels,
#         values=percentages,
#         textinfo='percent+label',
#         marker=dict(colors=colors),
#         hoverinfo='skip',
#         hole=0.3,
#     ))

#     # Add title and update layout
#     fig.update_layout(title_text="Percentage of Data", showlegend=False)

#     # Create the animation frames
#     frames = [go.Frame(data=[go.Pie(
#         labels=labels,
#         values=np.roll(percentages, shift),
#         textinfo='percent+label',
#         marker=dict(colors=colors),
#         hoverinfo='skip',
#         hole=0.3,
#     )], name=str(shift)) for shift in range(0, 360, 5)]

#     fig.update(frames=frames)

#     # Define animation settings
#     animation_settings = dict(frame=dict(duration=100, redraw=True), fromcurrent=True)

#     # Add play and pause buttons
#     fig.update_layout(updatemenus=[dict(type='buttons', showactive=False, buttons=[dict(label='Play',
#                                                                                       method='animate',
#                                                                                       args=[None,
#                                                                                             animation_settings]),
#                                                                                  dict(label='Pause',
#                                                                                       method='animate',
#                                                                                       args=[[None],
#                                                                                             animation_settings])])])

#     # Show the figure
#     fig.show()

# if __name__ == "__main__":
#     # Sample data - replace with your actual data
#     labels = ["Label A", "Label B", "Label C", "Label D"]
#     percentages = [25, 30, 20, 25]

#     create_animated_pie_chart(labels, percentages)


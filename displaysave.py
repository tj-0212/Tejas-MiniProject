import matplotlib.pyplot as plt

def visualize_percentage(labels, percentages, save_path=None):
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

    # Save the chart if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    # Show or save the chart depending on save_path
    if save_path:
        plt.close()  # Close the figure if saving to file
    else:
        plt.show()

if __name__ == "__main__":
    # Sample data - replace with your actual data
    labels = ["Label A", "Label B", "Label C", "Label D"]
    percentages = [25, 30, 20, 25]

    save_path = "pie_chart.png"  # Provide the desired path and filename to save the chart
    visualize_percentage(labels, percentages, save_path)

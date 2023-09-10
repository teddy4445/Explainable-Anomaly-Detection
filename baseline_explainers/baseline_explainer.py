import matplotlib.pyplot as plt


class BaselineExplainer:
    def __init__(self, data, model, mode):
        self.data = data
        self.features = list(self.data.columns.values)
        self.model = model
        self.mode = mode

    def get_explanation(self, anomaly):
        pass

    @staticmethod
    def save_barplot_explanation(path, name, show=True):
        # 1. Retrieve the Figure and Axes
        fig, ax = plt.gcf(), plt.gca()

        # 2. Get Text Elements (tick labels in this case)
        tick_labels = ax.get_yticklabels()

        # Minimum readable font size
        min_font_size = 12

        # Initialize required axis width
        required_axis_width = ax.get_position().width

        # 3. Calculate Required Width for each label
        for label in tick_labels:
            label.set_fontsize(min_font_size)  # Set to minimum readable font size
            text_width = label.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi  # in inches
            required_text_space = text_width / fig.get_size_inches()[0]  # Convert to relative figure coordinates
            required_axis_width = max(required_axis_width,
                                      1 - (ax.get_position().x0 + required_text_space))  # Update required axis width

        # 4. Adjust Axes Width
        ax.set_position([ax.get_position().x0, ax.get_position().y0, required_axis_width, ax.get_position().height])

        # 5. Set Font Size for all labels
        for label in tick_labels:
            label.set_fontsize(min_font_size)

        # 6. Adjust layout
        plt.tight_layout()

        # 7. Save or display the figure
        plt.savefig(f'{path}/{name}.png')
        if show:
            plt.show()

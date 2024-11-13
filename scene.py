from manimlib import *


class GraphExample(InteractiveScene):
    def construct(self):

        x_max = 10
        y_max = 0.5

        axes = Axes((0, x_max), (0, y_max, 0.1), dict(), dict(), {
            'unit_size': x_max / y_max * 0.5,
            'decimal_number_config': dict(
                num_decimal_places=1,
                font_size=36,
            )
        })
        axes.add_coordinate_labels()

        self.play(Write(axes, lag_ratio=0.01, run_time=1))

        train_err_bezier = bezier([0.4, 0.1, 0.05])
        train_err_fn = lambda x: train_err_bezier((x-1)/.9 / x_max)

        train_err_graph = axes.get_graph(
            train_err_fn,
            x_range=(1, 10),
            color=BLUE,
        )
        
        train_label = axes.get_graph_label(train_err_graph, "e_{train}")

        # Show training error
        self.play(
            ShowCreation(train_err_graph),
            FadeIn(train_label, RIGHT),
        )
        self.wait(1)

        test_err_bezier = bezier([0.45, 0.2, 0.4])
        val_err_fn = lambda x: test_err_bezier((x-1)/.9 / x_max)

        val_err_graph = axes.get_graph(
            val_err_fn,
            x_range=(1, 10),
            color=YELLOW,
        )
        
        # Show validation error
        val_label = axes.get_graph_label(val_err_graph, "e_{val}")
        self.play(
            ShowCreation(val_err_graph),
            FadeIn(val_label, RIGHT),
        )
        self.wait(1)

        # Show overfitting and underfitting
        over_under_line = DashedLine(
            axes.coords_to_point(x_max / 2, -y_max * 0.2), 
            axes.coords_to_point(x_max / 2, y_max * 1.2))
        self.play(ShowCreation(over_under_line))

        self.wait(1)

        # Show error arrows for overfitting
        arrow_x = x_max * 0.8
        arrow_start_y = \
            (train_err_fn(arrow_x) + 
            val_err_fn(arrow_x)) / 2
        overfitting_arrows = VGroup (Arrow(
            axes.coords_to_point(arrow_x, arrow_start_y),
            axes.coords_to_point(arrow_x, train_err_fn(arrow_x))
        ), Arrow(
            axes.coords_to_point(arrow_x, arrow_start_y),
            axes.coords_to_point(arrow_x, val_err_fn(arrow_x))
        ))
        self.play(FadeIn(overfitting_arrows))

        # Show label for overfitting
        label_y = arrow_start_y
        overfitting_label = Text("Overfitting", color=RED_B, font_size=36)
        overfitting_label.next_to(axes.coords_to_point(arrow_x, label_y), LEFT)
        self.play(FadeIn(overfitting_label))

        # Show error arrows for underfitting
        arrow_x = x_max * 0.2
        arrow_start_y = y_max * 0.15
        underfitting_arrows = VGroup (Arrow(
            axes.coords_to_point(arrow_x, arrow_start_y),
            axes.coords_to_point(arrow_x, train_err_fn(arrow_x))
        ), Arrow(
            axes.coords_to_point(arrow_x * 1.1, arrow_start_y),
            axes.coords_to_point(arrow_x * 1.1, val_err_fn(arrow_x))
        ))
        self.play(FadeIn(underfitting_arrows))

        # Show label for underfitting
        underfitting_label = Text("Underfitting", color=RED_D, font_size=36)
        underfitting_label.next_to(axes.coords_to_point(arrow_x * 1.1, label_y), RIGHT)
        self.play(FadeIn(underfitting_label))
        self.wait(1)

        # Hide overfitting and underfitting and training error
        self.play(FadeOut(VGroup(
            over_under_line, 
            overfitting_label, 
            underfitting_label,
            overfitting_arrows,
            underfitting_arrows,
        )))
        self.wait(1)

        self.play(FadeOut(VGroup(train_err_graph, train_label)))
        self.wait(1)

        # Show validation samples
        points = VGroup()
        n_points = 3
        for i in range(n_points):
            x = x_max / (n_points + 2) * (i + 2)
            y = val_err_fn(x)
            point = Dot(axes.coords_to_point(x, y))
            point.set_fill(YELLOW)
            point.set_stroke(GREY, 1)
            points.add(point)

        self.play(ShowCreation(points), run_time=3)
        self.wait(1)

        # Semi-hide validation error graph
        self.play(ApplyMethod(val_err_graph.set_stroke, {'opacity': 0.2}))
        self.wait(1)

        # Highlight an individual point
        point = points[0]
        highlight_rect = RoundedRectangle(height=0.5, width=0.5, corner_radius=0.1)
        highlight_rect.move_to(point)
        highlight_rect.set_stroke(WHITE)
        self.play(
            ShowCreation(highlight_rect),
            points[1:].animate.set_opacity(0.2),
        )
        self.wait(1)

        # Fade out the rect
        self.play(
            point.animate.set_opacity(0.2),
            FadeOut(highlight_rect),
        )

        # Show the model creation
        model = RoundedRectangle(height=1.5, width=1.5, corner_radius=0.1)
        model.move_to(axes.coords_to_point(x_max * 0.3, y_max * 0.3))
        model.set_stroke(GREEN)
        model_label = Text("Model", color=model.color, font_size=36)
        model_label.move_to(model.get_center())
        model_label.shift(0.3 * UP)

        self.play(ShowCreation(VGroup(model, model_label)))
        self.wait(1)

        # Dashed line from point to value on x-axis
        hp_value = 4
        x_line = DashedLine(
            axes.coords_to_point(hp_value, 0),
            highlight_rect.get_bottom(),
            color=WHITE,
        )

        # Highlight the value on the x-axis
        hp_value_axis_number = axes.get_axes()[0].numbers[3]
        hp_value_highlight = RoundedRectangle(hp_value_axis_number.get_width() + 0.2, hp_value_axis_number.get_height() + 0.2, corner_radius=0.1)
        hp_value_highlight.move_to(hp_value_axis_number)
        self.play(
            ShowCreation(hp_value_highlight),
        )
        hp_value_axis_number.set_color(RED_C)

        tex_color_map = {
            # 'X': TEAL,
            # 'y': TEAL,
            'val': GREY_B,
        }

        # Show the model hyperparameter
        hp_label = Tex(f"k = {hp_value}", font_size=24, color=WHITE)
        hp_label.next_to(model_label, DOWN)
        hp_label.set_color_by_tex_to_color_map(tex_color_map)

        hp_value_label = hp_label[f"{hp_value}"]
        hp_value_label.set_color(RED_C)

        self.play(FadeIn(hp_label))
        self.wait(1)

        # Show the input data label
        training_data_label = Tex("X_{val}", font_size=28, color=WHITE)
        training_data_label.next_to(model, LEFT)
        training_data_label.shift(0.8 * LEFT)
        training_data_label.set_color_by_tex_to_color_map(tex_color_map)

        # Arrow from training data to model
        arrow = Arrow(
            training_data_label.get_right(),
            model.get_left(),
            color=WHITE,
        )
        self.play(FadeIn(VGroup(arrow, training_data_label)))
        self.wait(1)


        # Show the prediction arrow and label
        predictions_label = Tex("y'_{val}", font_size=28, color=WHITE)
        predictions_label.next_to(model, RIGHT)
        predictions_label.shift(0.8 * RIGHT)
        predictions_label.set_color_by_tex_to_color_map(tex_color_map)

        # Arrow from training data to model
        arrow = Arrow(
            model.get_right(),
            predictions_label.get_left(),
            color=WHITE,
        )
        self.play(FadeIn(VGroup(arrow, predictions_label)))
        self.wait(1)



        # Error label
        error = f"{val_err_fn(hp_value):.2f}"
        error_label = Tex("E(y_{val}, y'_{val}) = " + error, font_size=28, color=WHITE)
        error_label.move_to(axes.coords_to_point(x_max * 0.7, y_max * 0.3))
        error_label.set_color_by_tex_to_color_map(tex_color_map)
        error_value = error_label[error]
        error_value.set_color(BLUE_C)

        self.play(
            TransformMatchingTex(predictions_label.copy(), error_label)
        )

        # Highlight the error value
        self.play(error_value.animate.shift(0.1 * RIGHT), run_time=0.3)
        error_highlight = RoundedRectangle(error_value.get_width() + 0.2, error_value.get_height() + 0.2, corner_radius=0.1)
        error_highlight.move_to(error_value)
        error_highlight.set_fill(BLACK)
        self.play(ShowCreation(error_highlight))
        self.wait(1)

        # Move the error value to it's value position
        error_value_group = VGroup(error_value.copy(), error_highlight)
        self.play(error_value_group.animate.move_to(
            axes.coords_to_point(-0.5, 0)
        ))
        self.wait(1)
        self.play(error_value_group.animate.move_to(
            axes.coords_to_point(-0.5, val_err_fn(hp_value))
        ))
        self.wait(1)

        # Dashed line from error value to x-axis
        y_line = DashedLine(
            axes.coords_to_point(0, val_err_fn(hp_value)),
            highlight_rect.get_left(),
        )
        self.play(
            ShowCreation(y_line),
            ShowCreation(x_line),
        )
        self.play(point.animate.set_opacity(1))
        self.wait(1)

        # Fade out the labels
        self.play(
            FadeOut(y_line),
            FadeOut(x_line),
        )
        self.wait(1)

        # Remove specific point highlights
        self.play(FadeOut(VGroup(
            error_value_group,
        )))
        hp_value_axis_number.set_color(WHITE)
        self.wait(1)
        
        # Update HP value
        hp_value = 6
        next_hp_label = Tex(f"k = {hp_value}", font_size=24)
        next_hp_label.move_to(hp_label)
        next_hp_label[f"{hp_value}"].set_color(RED_C)
        self.play(
            TransformMatchingTex(hp_label, next_hp_label),
            hp_value_highlight.animate.move_to(axes.get_axes()[0].numbers[hp_value - 1]),
        )

        # Update error value
        error = f"{val_err_fn(hp_value):.2f}"
        next_error_label = Tex("E(y_{val}, y'_{val}) = " + error, font_size=28, color=WHITE)
        next_error_label.set_color_by_tex_to_color_map(tex_color_map)
        next_error_label.move_to(error_label)
        next_error_label[error].set_color(BLUE_C)
        self.play(TransformMatchingTex(error_label, next_error_label))

        # Move coordinates
        coordinates = VGroup(
            next_error_label[error].copy(),
            next_hp_label[f"{hp_value}"].copy(),
        )
        self.play(
            ApplyMethod(coordinates[0].move_to, points[1]),
            ApplyMethod(coordinates[1].move_to, points[1]),
        )
        self.play(
            ApplyMethod(coordinates[0].set_opacity, 0),
            ApplyMethod(coordinates[1].set_opacity, 0),
            points[1].animate.set_opacity(1),
        )
        hp_label = next_hp_label
        error_label = next_error_label

        # Update HP value again
        hp_value = 8
        next_hp_label = Tex(f"k = {hp_value}", font_size=24)
        next_hp_label.move_to(hp_label)
        next_hp_label[f"{hp_value}"].set_color(RED_C)
        self.play(
            TransformMatchingTex(hp_label, next_hp_label),
            hp_value_highlight.animate.move_to(axes.get_axes()[0].numbers[hp_value - 1]),
        )

        # Update error value again
        error = f"{val_err_fn(hp_value):.2f}"
        next_error_label = Tex("E(y_{val}, y'_{val}) = " + error, font_size=28, color=WHITE)
        next_error_label.set_color_by_tex_to_color_map(tex_color_map)
        next_error_label.move_to(error_label)
        next_error_label[error].set_color(BLUE_C)
        self.play(TransformMatchingTex(error_label, next_error_label))

        # Move coordinates again
        coordinates = VGroup(
            next_error_label[error].copy(),
            next_hp_label[f"{hp_value}"].copy(),
        )
        self.play(
            ApplyMethod(coordinates[0].move_to, points[2]),
            ApplyMethod(coordinates[1].move_to, points[2]),
        )
        self.play(
            ApplyMethod(coordinates[0].set_opacity, 0),
            ApplyMethod(coordinates[1].set_opacity, 0),
            points[2].animate.set_opacity(1),
        )

        self.embed()
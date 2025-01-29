from manimlib import *

class SelectingPoints(InteractiveScene):
    def construct(self):

        # Construct X-Y axes
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

        self.play(
            ShowCreation(val_err_graph),
        )
        self.wait(1)


class ValPoints(InteractiveScene):

    def on_mouse_press(self, point, button, mods):
        if button == 1:
            self.go_to_next = True
        return super().on_mouse_press(point, button, mods)

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol == 32: # space
            self.go_to_next = True
        return super().on_key_press(symbol, modifiers)

    def wait_for_next(self):
        if self.skip_animations:
            return
        self.go_to_next = False
        self.wait(duration=1000, stop_condition=lambda: self.go_to_next)

    def construct(self):
        self.go_to_next = False

        # Construct X-Y axes
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
        self.wait_for_next()

        # Show labels
        x_axis_label = Text("Hyperparameter Value", font_size=24)
        x_axis_label.next_to(axes.get_bottom(), DOWN)
        x_axis_label.set_color(GREY_B)
        self.play(FadeIn(x_axis_label))
        self.wait_for_next()

        y_axis_label = Tex("e", font_size=36)
        y_axis_label.next_to(axes.get_left(), LEFT)
        y_axis_label.set_color(GREY_B)
        self.play(FadeIn(y_axis_label))
        self.wait_for_next()

        # Add model complexity
        added_label = Text("(Model Complexity)", font_size=24)
        added_label.next_to(x_axis_label, DOWN)
        added_label.set_color(x_axis_label.get_color())
        self.play(FadeIn(added_label))
        self.wait_for_next()


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
        self.wait_for_next()

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
        self.wait_for_next()

        # Show overfitting and underfitting
        over_under_line = DashedLine(
            axes.coords_to_point(x_max / 2, y_max * 1),
            axes.coords_to_point(x_max / 2, -y_max * 0))
        self.play(ShowCreation(over_under_line))

        self.wait_for_next()

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
        underfitting_label.shift(0.8 * DOWN)
        self.play(FadeIn(underfitting_label))
        self.wait_for_next()

        # Hide overfitting and underfitting and training error
        self.play(FadeOut(VGroup(
            over_under_line,
            overfitting_label,
            underfitting_label,
            overfitting_arrows,
            underfitting_arrows,
        )))
        self.wait_for_next()

        # Animate a minimum-finding point
        roll_x = lambda alpha: math.cos(alpha * PI * 3) * 2.5 * math.pow(1-alpha, 1.2) + 5.5
        min_point = Dot()
        min_point.move_to(axes.coords_to_point(roll_x(0), val_err_fn(roll_x(0))))
        min_point.set_fill(YELLOW_E)
        min_point.set_stroke(GREY, 1)
        self.play(FadeIn(min_point))

        class RollAnimation(Animation):
            def __init__(self, point: Dot, **kwargs) -> None:
                # Pass number as the mobject of the animation
                super().__init__(point,  **kwargs)

            def interpolate_mobject(self, alpha: float) -> None:
                x = roll_x(alpha)
                y = val_err_fn(x)
                self.mobject.move_to(axes.coords_to_point(x, y))
        self.play(RollAnimation(min_point), run_time=4, rate_func=linear)
        self.wait_for_next()

        # Fade out the minimum point
        self.play(FadeOut(min_point))

        # Fade out training error plot
        self.play(FadeOut(VGroup(train_err_graph, train_label)))
        self.wait_for_next()

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
        self.wait_for_next()

        # Semi-hide validation error graph
        self.play(ApplyMethod(val_err_graph.set_stroke, {'opacity': 0.2}))
        self.wait_for_next()

        # Highlight an individual point
        point = points[0]
        highlight_rect = RoundedRectangle(height=0.5, width=0.5, corner_radius=0.1)
        highlight_rect.move_to(point)
        highlight_rect.set_stroke(WHITE)
        self.play(
            ShowCreation(highlight_rect),
            points[1:].animate.set_opacity(0.2),
        )
        self.wait_for_next()

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
        self.wait_for_next()

        # Dashed line from point to value on x-axis
        hp_value = 4
        x_line = DashedLine(
            axes.coords_to_point(hp_value, 0),
            point.get_bottom(),
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
            'train': GREY_B,
        }

        # Show the model hyperparameter
        hp_label = Tex(f"k = {hp_value}", font_size=24, color=WHITE)
        hp_label.next_to(model_label, DOWN)
        hp_label.set_color_by_tex_to_color_map(tex_color_map)

        hp_value_label = hp_label[f"{hp_value}"]
        hp_value_label.set_color(RED_C)

        self.play(FadeIn(hp_label))
        self.wait_for_next()

        # Show training data
        def show_training(model):
            training_data = Tex("(X_{train}, y_{train})", font_size=28, color=WHITE)
            training_data.next_to(model, UP)
            training_data.shift(0.5 * UP)
            training_arrow = Arrow(
                training_data.get_bottom(),
                model.get_top(),
                color=WHITE,
            )

            self.play(FadeIn(VGroup(training_data, training_arrow)))
            self.play(
                VGroup(training_data, training_arrow).animate.move_to(model).set_width(0, False),
                model.animate.set_fill(GREEN_E, opacity=0.3),
            )
            self.play(model.animate.set_fill(opacity=0), run_time=0.5)
        show_training(model)

        # Show the input data label
        val_data_label = Tex("X_{val}", font_size=28, color=WHITE)
        val_data_label.next_to(model, LEFT)
        val_data_label.shift(0.8 * LEFT)
        val_data_label.set_color_by_tex_to_color_map(tex_color_map)

        # Arrow from validation data to model
        arrow_val = Arrow(
            val_data_label.get_right(),
            model.get_left(),
            color=WHITE,
        )
        self.play(FadeIn(VGroup(arrow_val, val_data_label)))
        self.wait_for_next()


        # Show the prediction arrow and label
        predictions_label = Tex("y'_{val}", font_size=28, color=WHITE)
        predictions_label.next_to(model, RIGHT)
        predictions_label.shift(0.8 * RIGHT)
        predictions_label.set_color_by_tex_to_color_map(tex_color_map)

        # Arrow from model to predictions
        arrow_pred = Arrow(
            model.get_right(),
            predictions_label.get_left(),
            color=WHITE,
        )
        self.play(FadeIn(VGroup(arrow_pred, predictions_label)))
        self.wait_for_next()



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
        self.wait_for_next()

        # Move the error value to it's value position
        error_value_group = VGroup(error_value.copy(), error_highlight)
        self.play(error_value_group.animate.move_to(
            axes.coords_to_point(-0.5, 0)
        ))
        self.wait_for_next()
        self.play(error_value_group.animate.move_to(
            axes.coords_to_point(-0.5, val_err_fn(hp_value))
        ))
        self.wait_for_next()

        # Dashed line from error value to x-axis
        y_line = DashedLine(
            axes.coords_to_point(0, val_err_fn(hp_value)),
            point.get_left(),
        )
        self.play(
            ShowCreation(y_line),
            ShowCreation(x_line),
        )
        self.play(point.animate.set_opacity(1))
        self.wait_for_next()

        # Fade out the labels
        self.play(
            FadeOut(y_line),
            FadeOut(x_line),
        )
        self.wait_for_next()

        # Remove specific point highlights
        self.play(FadeOut(VGroup(
            error_value_group,
        )))
        hp_value_axis_number.set_color(WHITE)
        self.wait_for_next()

        # Update HP value
        hp_value = 6
        next_hp_label = Tex(f"k = {hp_value}", font_size=24)
        next_hp_label.move_to(hp_label)
        next_hp_label[f"{hp_value}"].set_color(RED_C)
        self.play(
            hp_value_highlight.animate.move_to(axes.get_axes()[0].numbers[hp_value - 1]),
        )
        self.play(
            TransformMatchingTex(hp_label, next_hp_label),
            run_time=0.5,
        )

        show_training(model)

        # Update error value
        error = f"{val_err_fn(hp_value):.2f}"
        next_error_label = Tex("E(y_{val}, y'_{val}) = " + error, font_size=28, color=WHITE)
        next_error_label.set_color_by_tex_to_color_map(tex_color_map)
        next_error_label.move_to(error_label)
        next_error_label[error].set_color(BLUE_C)
        self.play(TransformMatchingTex(error_label, next_error_label), run_time=0.5)
        self.wait_for_next()

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
            hp_value_highlight.animate.move_to(axes.get_axes()[0].numbers[hp_value - 1]),
        )
        self.play(
            TransformMatchingTex(hp_label, next_hp_label),
            run_time=0.5,
        )

        show_training(model)

        # Update error value again
        error = f"{val_err_fn(hp_value):.2f}"
        next_error_label = Tex("E(y_{val}, y'_{val}) = " + error, font_size=28, color=WHITE)
        next_error_label.set_color_by_tex_to_color_map(tex_color_map)
        next_error_label.move_to(error_label)
        next_error_label[error].set_color(BLUE_C)
        self.play(TransformMatchingTex(error_label, next_error_label), run_time=0.5)
        self.wait_for_next()

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
        self.wait_for_next()
        hp_label = next_hp_label
        error_label = next_error_label

        # Fade out most things
        self.play(FadeOut(VGroup(
            model,
            model_label,
            hp_label,
            val_data_label,
            arrow_val,
            arrow_pred,
            predictions_label,
            error_label,
            hp_value_highlight,
        )))
        self.wait_for_next()

        # Highlight the minimum point
        highlight_rect.move_to(points[1])
        self.play(ShowCreation(highlight_rect))
        self.wait_for_next()

        self.embed()

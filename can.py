from manim import *

class CandleLightingAnimation(Scene):
    def construct(self):
        # Create the candle
        candle_body = Rectangle(
            height=3.0,
            width=0.8,
            fill_color=WHITE,
            fill_opacity=1,
            stroke_color=GRAY,
            stroke_width=2
        ).move_to(DOWN)
        
        # Create the wick
        wick = Line(
            start=candle_body.get_top(),
            end=candle_body.get_top() + UP * 0.3,
            stroke_color=BLACK,
            stroke_width=4
        )
        
        # Create the flame (initially invisible)
        flame = VMobject()
        flame_points = [
            UP * 0.6,
            RIGHT * 0.3 + UP * 0.3,
            RIGHT * 0.1 + DOWN * 0.1,
            LEFT * 0.1 + DOWN * 0.1,
            LEFT * 0.3 + UP * 0.3
        ]
        flame.set_points_as_corners(flame_points)
        flame.set_fill(YELLOW, opacity=0)
        flame.set_stroke(RED, opacity=0, width=0)
        flame.move_to(wick.get_top() + UP * 0.2)
        
        # Create a background for contrast
        # Use config.frame_width and config.frame_height instead of FRAME_WIDTH and FRAME_HEIGHT
        background = Rectangle(
            width=config.frame_width,
            height=config.frame_height,
            fill_color=BLUE_E,
            fill_opacity=1
        )

        # Add objects to the scene
        self.add(background)
        self.add(candle_body, wick)
        
        # Animation for initial setup
        self.wait(1)
        
        # Animation for lighting the candle
        self.play(
            flame.animate.set_fill(YELLOW, opacity=1),
            flame.animate.set_stroke(RED, opacity=1, width=2),
            run_time=1.5
        )
        
        # Flame flickering animation
        for _ in range(3):
            self.play(
                flame.animate.scale(1.2).set_fill(color=YELLOW_A),
                run_time=0.5
            )
            self.play(
                flame.animate.scale(1/1.2).set_fill(color=YELLOW),
                run_time=0.5
            )
        
        # Create a subtle glow around the flame
        glow = Dot(
            radius=0.6,
            fill_color=YELLOW,
            fill_opacity=0.3
        ).move_to(flame.get_center())
        
        self.play(FadeIn(glow), run_time=1)
        
        # Make the flame and glow dance together
        for _ in range(2):
            self.play(
                glow.animate.scale(1.3).set_opacity(0.4),
                flame.animate.scale(1.1).shift(UP * 0.1),
                run_time=0.7
            )
            self.play(
                glow.animate.scale(1/1.3).set_opacity(0.3),
                flame.animate.scale(1/1.1).shift(DOWN * 0.1),
                run_time=0.7
            )
        
        # Slowly brighten the scene to simulate the candle lighting the area
        light_overlay = Circle(
            radius=5,
            fill_color=YELLOW,
            fill_opacity=0
        ).move_to(flame.get_center())
        
        self.play(
            light_overlay.animate.set_opacity(0.1),
            run_time=2
        )
        
        self.wait(2)
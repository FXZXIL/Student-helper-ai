from manim import *

class BoltExample(Scene):
    def bolt_new(self, shape_type):
        """ Simulates `bolt.new` by dynamically creating a shape. """
        if shape_type == "circle":
            return Circle()
        elif shape_type == "square":
            return Square()
        elif shape_type == "triangle":
            return Triangle()
        else:
            return Dot()  # Default shape

    def construct(self):
        shape1 = self.bolt_new("circle")  # Create a circle
        shape2 = self.bolt_new("square")  # Create a square
        shape3 = self.bolt_new("triangle")  # Create a triangle

        shape1.set_color(RED).shift(LEFT * 2)
        shape2.set_color(GREEN)
        shape3.set_color(BLUE).shift(RIGHT * 2)

        self.play(Create(shape1))  # Animate circle
        self.wait(1)
        self.play(Create(shape2))  # Animate square
        self.wait(1)
        self.play(Create(shape3))  # Animate triangle
        self.wait(2)

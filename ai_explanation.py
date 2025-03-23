from manim import *

class EnhancedAIExplanation(Scene):
    def construct(self):
        # Title with attractive animation
        title = Text("Understanding Artificial Intelligence", font_size=52, color=BLUE_D)
        subtitle = Text("From Concepts to Applications", font_size=32, color=YELLOW_D).next_to(title, DOWN)
        title_group = VGroup(title, subtitle)
        
        self.play(Write(title, run_time=1.5))
        self.play(FadeIn(subtitle, shift=DOWN))
        self.wait(1)
        self.play(title_group.animate.scale(0.6).to_edge(UP))
        
        # SECTION 1: WHAT IS AI?
        section_title = Text("What is Artificial Intelligence?", font_size=40, color=WHITE)
        self.play(Write(section_title))
        self.wait(0.5)
        self.play(FadeOut(section_title))
        
        # More detailed definition with hierarchical structure
        ai_definition = VGroup(
            Text("Artificial Intelligence:", font_size=36, color=YELLOW),
            Text("Technology that enables machines to simulate human intelligence", font_size=28, color=WHITE)
        ).arrange(DOWN, aligned_edge=LEFT).shift(UP * 2)
        
        ai_categories = VGroup(
            Text("Key AI Categories:", font_size=30, color=ORANGE),
            Dot(color=WHITE).scale(0.5).next_to(LEFT * 3 + DOWN * 0.5, RIGHT),
            Text("Narrow AI: Specialized at specific tasks", font_size=26, color=WHITE),
            Dot(color=WHITE).scale(0.5).next_to(LEFT * 3 + DOWN * 1.0, RIGHT),
            Text("General AI: Broad human-like capabilities", font_size=26, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(ai_definition, DOWN, buff=0.5)
        
        ai_categories[1].align_to(ai_categories[2], LEFT).shift(RIGHT * 0.5)
        ai_categories[3].align_to(ai_categories[4], LEFT).shift(RIGHT * 0.5)
        
        self.play(Write(ai_definition, run_time=2))
        self.play(Write(ai_categories, run_time=3))
        self.wait(2)
        self.play(FadeOut(ai_definition), FadeOut(ai_categories))
        
        # SECTION 2: HOW AI WORKS - NEURAL NETWORKS VISUALIZATION
        nn_title = Text("How AI Works: Neural Networks", font_size=38, color=BLUE).shift(UP * 3)
        self.play(Write(nn_title))
        
        # Create a more complex neural network
        network_scale = 0.8
        layer_distances = [4, 2.5, 1, -1, -2.5, -4]  # x-coordinates for layers
        layers_config = [4, 6, 8, 8, 6, 3]  # Number of neurons in each layer
        
        neurons = []
        connections = []
        layer_labels = []
        
        # Create the neural network visualization
        for layer_idx, (x_pos, num_neurons) in enumerate(zip(layer_distances, layers_config)):
            layer_neurons = []
            
            # Create neurons for this layer
            for i in range(num_neurons):
                y_pos = (i - (num_neurons - 1) / 2) * 0.5
                neuron = Circle(radius=0.15, color=BLUE_E, fill_opacity=0.8).scale(network_scale)
                neuron.move_to([x_pos * network_scale, y_pos * network_scale, 0])
                layer_neurons.append(neuron)
            
            neurons.append(layer_neurons)
            
            # Create connections to previous layer
            if layer_idx > 0:
                prev_layer = neurons[layer_idx - 1]
                for neuron in layer_neurons:
                    for prev_neuron in prev_layer:
                        connection = Line(
                            prev_neuron.get_center(),
                            neuron.get_center(),
                            color=GRAY,
                            stroke_opacity=0.7,
                            stroke_width=1
                        )
                        connections.append(connection)
            
            # Layer labels
            if layer_idx == 0:
                label = Text("Input Layer", font_size=20, color=GREEN_D)
            elif layer_idx == len(layers_config) - 1:
                label = Text("Output Layer", font_size=20, color=RED_D)
            else:
                label = Text(f"Hidden Layer {layer_idx}", font_size=20, color=BLUE_D)
            
            label.next_to(VGroup(*layer_neurons), DOWN, buff=0.7)
            layer_labels.append(label)
        
        # Flatten all neurons for animation
        all_neurons = [neuron for layer in neurons for neuron in layer]
        
        # Animate network creation
        self.play(*[FadeIn(neuron) for neuron in all_neurons], run_time=2)
        self.play(*[Create(conn, run_time=0.01) for conn in connections], run_time=3)
        self.play(*[Write(label) for label in layer_labels], run_time=2)
        
        # Signal propagation through the network
        signal_animations = []
        for layer_idx in range(len(layers_config) - 1):
            curr_layer = neurons[layer_idx]
            next_layer = neurons[layer_idx + 1]
            
            for start_neuron in curr_layer:
                for end_neuron in next_layer:
                    dot = Dot(color=YELLOW, radius=0.08).move_to(start_neuron.get_center())
                    signal_animations.append(
                        dot.animate.move_to(end_neuron.get_center()).set_color(ORANGE)
                    )
        
        self.play(AnimationGroup(*signal_animations[:20], lag_ratio=0.05), run_time=4)
        
        # Decision making animation
        decision_text = Text("Neural networks process input through layers to make predictions",
                           font_size=24, color=WHITE).shift(DOWN * 3)
        self.play(Write(decision_text))
        self.wait(2)
        
        # Clean up for next section
        self.play(
            FadeOut(nn_title),
            *[FadeOut(neuron) for neuron in all_neurons],
            *[FadeOut(conn) for conn in connections],
            *[FadeOut(label) for label in layer_labels],
            FadeOut(decision_text)
        )
        
        # SECTION 3: LEARNING PROCESS - TRAINING DATA AND ALGORITHMS
        learning_title = Text("How AI Learns: Training Process", font_size=38, color=GREEN).shift(UP * 3)
        self.play(Write(learning_title))
        
        # Training data visualization with improved graphics
        training_data = VGroup()
        
        # Class 1 data points (cats) - red cluster
        for _ in range(10):
            x = np.random.uniform(-4, -2)
            y = np.random.uniform(-1, 1)
            point = Dot(point=[x, y, 0], color=RED, radius=0.08)
            training_data.add(point)
        
        # Class 2 data points (dogs) - blue cluster
        for _ in range(10):
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(0, 2)
            point = Dot(point=[x, y, 0], color=BLUE, radius=0.08)
            training_data.add(point)
        
        data_frame = SurroundingRectangle(training_data, color=GRAY, buff=0.5)
        data_label = Text("Training Data: Cats (Red) vs Dogs (Blue)", font_size=24, color=WHITE)
        data_label.next_to(data_frame, DOWN)
        
        self.play(FadeIn(training_data), Create(data_frame), Write(data_label))
        
        # Decision boundary animation
        initial_boundary = Line(start=[-5, -2, 0], end=[2, 3, 0], color=YELLOW)
        boundary_label = Text("Initial Decision Boundary", font_size=20, color=YELLOW)
        boundary_label.next_to(initial_boundary, UP, buff=0.2)
        
        self.play(Create(initial_boundary), Write(boundary_label))
        
        # Show model improvement over iterations
        iterations_text = Text("Training over multiple iterations:", font_size=24, color=WHITE).shift(RIGHT * 4 + UP * 1)
        self.play(Write(iterations_text))
        
        # Different iterations of the model improving
        for i in range(3):
            new_boundary = Line(
                start=[-5, -2 + 0.5*i, 0],
                end=[2, 3 - 0.3*i, 0],
                color=YELLOW
            )
            iteration_label = Text(f"Iteration {i+1}", font_size=18, color=WHITE).shift(RIGHT * 4)
            
            if i == 0:
                iteration_label.next_to(iterations_text, DOWN)
            else:
                iteration_label.next_to(RIGHT * 4 + DOWN * (i * 0.5), RIGHT)
            
            self.play(
                Transform(initial_boundary, new_boundary),
                Transform(boundary_label, Text(f"Improved Boundary", font_size=20, color=YELLOW).next_to(new_boundary, UP, buff=0.2)),
                Write(iteration_label)
            )
            self.wait(0.5)
        
        loss_graph = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 5, 1],
            axis_config={"include_tip": False, "color": WHITE},
            x_axis_config={"label_direction": DOWN},
            y_axis_config={"label_direction": LEFT},
        ).scale(0.4).shift(RIGHT * 4 + DOWN * 1)
        
        loss_label = Text("Training Loss", font_size=20, color=WHITE).next_to(loss_graph, UP, buff=0.2)
        
        # Create decreasing loss curve
        loss_points = [5, 4, 3.3, 2.7, 2.2, 1.8, 1.5, 1.3, 1.2, 1.1]
        loss_dots = VGroup(*[
            Dot(loss_graph.coords_to_point(i, loss_points[i]), color=RED_E, radius=0.05)
            for i in range(len(loss_points))
        ])
        
        loss_line = VGroup(*[
            Line(
                loss_graph.coords_to_point(i, loss_points[i]),
                loss_graph.coords_to_point(i+1, loss_points[i+1]),
                color=RED_E
            )
            for i in range(len(loss_points)-1)
        ])
        
        self.play(Create(loss_graph), Write(loss_label))
        self.play(FadeIn(loss_dots), Create(loss_line, run_time=2))
        
        self.wait(2)
        
        # Clean up for next section
        self.play(
            FadeOut(learning_title),
            FadeOut(training_data),
            FadeOut(data_frame),
            FadeOut(data_label),
            FadeOut(initial_boundary),
            FadeOut(boundary_label),
            FadeOut(iterations_text),
            FadeOut(loss_graph),
            FadeOut(loss_dots),
            FadeOut(loss_line),
            FadeOut(loss_label)
        )
        
        # SECTION 4: REAL-WORLD APPLICATIONS
        applications_title = Text("AI Applications in Real World", font_size=38, color=PURPLE).shift(UP * 3)
        self.play(Write(applications_title))
        
        # Create application icons with descriptions
        app_icons = VGroup()
        app_descs = VGroup()
        
        # Healthcare application
        healthcare_icon = Circle(radius=0.8, color=RED_E, fill_opacity=0.8)
        healthcare_symbol = Text("+", font_size=50, color=WHITE).move_to(healthcare_icon.get_center())
        healthcare_label = Text("Healthcare", font_size=26, color=RED_E)
        healthcare_desc = Text("Disease detection & drug discovery", font_size=20, color=WHITE)
        
        # Transportation application
        transport_icon = Circle(radius=0.8, color=BLUE_E, fill_opacity=0.8)
        transport_symbol = Text("🚗", font_size=40, color=WHITE).move_to(transport_icon.get_center())
        transport_label = Text("Transportation", font_size=26, color=BLUE_E)
        transport_desc = Text("Self-driving vehicles & traffic optimization", font_size=20, color=WHITE)
        
        # Finance application
        finance_icon = Circle(radius=0.8, color=GREEN_E, fill_opacity=0.8)
        finance_symbol = Text("$", font_size=50, color=WHITE).move_to(finance_icon.get_center())
        finance_label = Text("Finance", font_size=26, color=GREEN_E)
        finance_desc = Text("Fraud detection & algorithmic trading", font_size=20, color=WHITE)
        
        # Position icons in a row
        icons_row = VGroup(
            VGroup(healthcare_icon, healthcare_symbol),
            VGroup(transport_icon, transport_symbol),
            VGroup(finance_icon, finance_symbol)
        ).arrange(RIGHT, buff=1.5).shift(UP * 0.5)
        
        # Position labels below icons
        healthcare_label.next_to(healthcare_icon, DOWN)
        transport_label.next_to(transport_icon, DOWN)
        finance_label.next_to(finance_icon, DOWN)
        
        # Position descriptions below labels
        healthcare_desc.next_to(healthcare_label, DOWN)
        transport_desc.next_to(transport_label, DOWN)
        finance_desc.next_to(finance_label, DOWN)
        
        # Add all elements to groups
        app_icons.add(
            healthcare_icon, healthcare_symbol,
            transport_icon, transport_symbol,
            finance_icon, finance_symbol
        )
        
        app_descs.add(
            healthcare_label, healthcare_desc,
            transport_label, transport_desc,
            finance_label, finance_desc
        )
        
        # Animate applications section
        self.play(FadeIn(app_icons, shift=UP))
        self.play(Write(app_descs))
        
        self.wait(2)
        
        # Clean up for conclusion
        self.play(
            FadeOut(applications_title),
            FadeOut(app_icons),
            FadeOut(app_descs)
        )
        
        # SECTION 5: FUTURE OF AI
        future_title = Text("The Future of AI", font_size=38, color=GOLD).shift(UP * 3)
        self.play(Write(future_title))
        
        # Create a futuristic timeline
        timeline = Line(start=LEFT * 4, end=RIGHT * 4, color=BLUE_C)
        current_marker = Dot(color=GREEN, radius=0.15).move_to(LEFT * 2)
        current_label = Text("Today", font_size=20, color=GREEN).next_to(current_marker, UP)
        
        # Future milestones
        milestones = VGroup(
            Dot(color=YELLOW, radius=0.15).move_to(RIGHT * 0),
            Dot(color=YELLOW, radius=0.15).move_to(RIGHT * 2),
            Dot(color=YELLOW, radius=0.15).move_to(RIGHT * 4)
        )
        
        milestone_labels = VGroup(
            Text("Advanced NLP", font_size=18, color=YELLOW).next_to(milestones[0], UP),
            Text("Human-level reasoning", font_size=18, color=YELLOW).next_to(milestones[1], UP),
            Text("General AI", font_size=18, color=YELLOW).next_to(milestones[2], UP)
        )
        
        milestone_years = VGroup(
            Text("Near future", font_size=16, color=WHITE).next_to(milestones[0], DOWN),
            Text("Mid-term", font_size=16, color=WHITE).next_to(milestones[1], DOWN),
            Text("Long-term", font_size=16, color=WHITE).next_to(milestones[2], DOWN)
        )
        
        self.play(Create(timeline))
        self.play(FadeIn(current_marker), Write(current_label))
        self.play(FadeIn(milestones), Write(milestone_labels), Write(milestone_years))
        
        # Ethical considerations
        ethics_title = Text("Ethical Considerations", font_size=28, color=RED).shift(DOWN * 1)
        self.play(Write(ethics_title))
        
        ethics_points = VGroup(
            Text("• Bias & Fairness", font_size=24, color=WHITE),
            Text("• Privacy Concerns", font_size=24, color=WHITE),
            Text("• Human-AI Collaboration", font_size=24, color=WHITE)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(ethics_title, DOWN)
        
        self.play(Write(ethics_points, run_time=2))
        
        self.wait(2)
        
        # Final cleanup before conclusion
        self.play(
            FadeOut(future_title),
            FadeOut(timeline),
            FadeOut(current_marker),
            FadeOut(current_label),
            FadeOut(milestones),
            FadeOut(milestone_labels),
            FadeOut(milestone_years),
            FadeOut(ethics_title),
            FadeOut(ethics_points)
        )
        
        # CONCLUSION SECTION
        conclusion = Text("AI is transforming our world through data and algorithms", 
                         font_size=36, color=YELLOW).shift(UP)
        subconclusion = Text("Understanding its principles helps us navigate this new era", 
                           font_size=28, color=WHITE).next_to(conclusion, DOWN)
        
        self.play(Write(conclusion))
        self.play(FadeIn(subconclusion, shift=UP))
        
        # Final animation - zoom out of everything
        final_group = VGroup(title_group, conclusion, subconclusion)
        self.play(final_group.animate.scale(0.8).shift(UP * 0.5))
        
        thanks = Text("Thanks for watching!", font_size=48, color=BLUE).shift(DOWN * 2)
        self.play(Write(thanks))
        
        self.wait(2)
        
        # Fade everything out
        self.play(FadeOut(final_group), FadeOut(thanks))
        self.wait(1)
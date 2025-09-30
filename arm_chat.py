#!/usr/bin/env python3
"""
ARM Chat Application

A standalone chat interface that:
1. Loads pre-saved ARM manifold analysis results
2. Uses manifold steering for text generation
3. Maintains conversation history
4. Supports long-form generation with context
"""

import gradio as gr
import json
import pickle
import torch
from typing import List, Tuple, Optional
from pathlib import Path

from arm_library.core.arm_mapper import ARMMapper
from arm_library.utils.config import ARMConfig


class ARMChatApp:
    """Chat application with ARM-steered generation."""
    
    def __init__(self):
        self.mapper: Optional[ARMMapper] = None
        self.manifold_data: Optional[dict] = None
        self.config: Optional[ARMConfig] = None
        self.conversation_history: List[Tuple[str, str]] = []
        
    def load_manifold(self, file_path: str) -> str:
        """Load a saved ARM manifold file."""
        try:
            file_path = Path(file_path)
            
            # Determine file type and load
            if file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
            elif file_path.suffix in ['.pkl', '.pickle']:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                return f"❌ Unsupported file type: {file_path.suffix}"
            
            # Extract configuration and results
            self.config = ARMConfig.from_dict(data['configuration'])
            self.manifold_data = self._convert_manifold_from_saved(data['results'])
            
            # Restore prompts into manifold data
            if 'prompts' in data:
                self.manifold_data['prompts'] = data['prompts']
            
            # Initialize mapper for generation
            self.mapper = ARMMapper(self.config)
            
            # Clear conversation history on new manifold load
            self.conversation_history = []
            
            status = f"✅ Manifold loaded successfully!\n"
            status += f"Model: {self.config.model_name}\n"
            status += f"Seeds: {len(self.manifold_data.get('prompts', []))}\n"
            status += f"Layer: {self.config.layer_to_probe}\n"
            
            return status
            
        except Exception as e:
            return f"❌ Failed to load manifold: {str(e)}"
    
    def _convert_manifold_from_saved(self, saved_data: dict) -> dict:
        """Convert saved manifold data back to working format."""
        import numpy as np
        
        results = {}
        for key, value in saved_data.items():
            if key == 'descriptors' and isinstance(value, list):
                results[key] = np.array(value)
            elif key == 'graph_data' and isinstance(value, dict):
                graph_data = value.copy()
                if 'feature_vectors' in graph_data:
                    graph_data['feature_vectors'] = np.array(graph_data['feature_vectors'])
                if 'spectral_embedding' in graph_data:
                    graph_data['spectral_embedding'] = np.array(graph_data['spectral_embedding'])
                if 'adjacency_matrix' in graph_data:
                    graph_data['adjacency_matrix'] = np.array(graph_data['adjacency_matrix'])
                results[key] = graph_data
            elif isinstance(value, dict):
                results[key] = self._convert_manifold_from_saved(value)
            else:
                results[key] = value
        
        return results
    
    def chat(
        self,
        user_message: str,
        steering_mode: str,
        target_signature_indices: str,
        steering_strength: float,
        max_tokens: int,
        temperature: float,
        history: List[Tuple[str, str]],
        basis_mode: str = "top1",
        single_mode_index: int = 0,
        blend_mode_indices: str = "",
        blend_mode_weights: str = "",
    ) -> Tuple[List[Tuple[str, str]], str]:
        """
        Generate a response to user message with ARM steering.
        
        Returns:
            Updated history and empty string for input box
        """
        if not self.mapper or not self.manifold_data:
            history.append((user_message, "⚠️ Please load a manifold file first!"))
            return history, ""
        
        if not user_message.strip():
            return history, ""
        
        try:
            # Build context from conversation history
            context = self._build_context(history, user_message)
            
            # Generate response with ARM steering
            if steering_mode == "none":
                # No steering - just generate
                generator = self.mapper.create_controlled_generator()
                response = generator.generate_with_steering(
                    context,
                    max_length=max_tokens,
                    temperature=temperature,
                    do_sample=True
                )
            elif steering_mode == "manifold_signature":
                # Steer toward blended signature(s)
                import numpy as np
                
                # Parse target indices
                try:
                    indices = [int(x.strip()) for x in target_signature_indices.split(',') if x.strip()]
                except ValueError:
                    response = "❌ Invalid indices format. Use comma-separated numbers (e.g., '0,1,2')"
                    history.append((user_message, response))
                    return history, ""
                
                if not indices:
                    response = "❌ Need at least one signature index"
                    history.append((user_message, response))
                    return history, ""
                
                # Validate and collect signatures
                max_index = len(self.manifold_data['seed_analyses']) - 1
                signatures = []
                for idx in indices:
                    if idx < 0 or idx > max_index:
                        response = f"❌ Index {idx} out of range (0-{max_index})"
                        history.append((user_message, response))
                        return history, ""
                    
                    target_analysis = self.manifold_data['seed_analyses'][idx]
                    signatures.append(target_analysis['resonance_signature']['s_norm'])
                
                # Blend signatures if multiple
                blended_signature = np.mean(signatures, axis=0) if len(signatures) > 1 else signatures[0]
                
                # Parse basis selection
                mode_indices = None
                mode_weights = None
                if basis_mode == "single":
                    mode_indices = [int(single_mode_index)]
                elif basis_mode == "blend":
                    try:
                        mode_indices = [int(x.strip()) for x in blend_mode_indices.split(',') if x.strip()]
                    except ValueError:
                        response = "❌ Invalid blend mode indices (use comma-separated integers)."
                        history.append((user_message, response))
                        return history, ""
                    if not mode_indices:
                        response = "❌ Provide at least one mode index for blend"
                        history.append((user_message, response))
                        return history, ""
                    if blend_mode_weights.strip():
                        try:
                            mode_weights = [float(x.strip()) for x in blend_mode_weights.split(',') if x.strip()]
                        except ValueError:
                            response = "❌ Invalid blend weights (use comma-separated numbers)."
                            history.append((user_message, response))
                            return history, ""
                        if len(mode_weights) != len(mode_indices):
                            response = "❌ Number of weights must match number of mode indices"
                            history.append((user_message, response))
                            return history, ""

                response = self.mapper.steer_generation_toward_signature(
                    prompt=context,
                    target_signature=blended_signature,
                    max_length=max_tokens,
                    temperature=temperature,
                    steering_strength=steering_strength,
                    mode_indices=mode_indices,
                    mode_weights=mode_weights,
                )
            else:
                response = "❌ Unknown steering mode"
            
            # Extract just the new generated text (remove context)
            response = response[len(context):].strip()
            
            # Update history
            history.append((user_message, response))
            
            return history, ""
            
        except Exception as e:
            history.append((user_message, f"❌ Generation failed: {str(e)}"))
            return history, ""
    
    def _build_context(self, history: List[Tuple[str, str]], current_message: str) -> str:
        """Build conversation context from history."""
        context_parts = []
        
        # Include recent conversation history (last 5 exchanges)
        recent_history = history[-5:] if len(history) > 5 else history
        
        for user_msg, bot_msg in recent_history:
            context_parts.append(f"User: {user_msg}")
            context_parts.append(f"Assistant: {bot_msg}")
        
        # Add current message
        context_parts.append(f"User: {current_message}")
        context_parts.append("Assistant:")
        
        return "\n".join(context_parts)
    
    def clear_history(self) -> List[Tuple[str, str]]:
        """Clear conversation history."""
        self.conversation_history = []
        return []


def create_chat_interface():
    """Create the Gradio chat interface."""
    
    app = ARMChatApp()
    
    with gr.Blocks(title="ARM Chat", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("""
        # 🗨️ ARM Chat
        
        **Load a pre-saved ARM manifold and chat with steered generation**
        
        1. Load your manifold file (from ARM Analysis save)
        2. Configure steering settings
        3. Start chatting!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📂 Load Manifold")
                
                manifold_file = gr.File(
                    label="ARM Manifold File",
                    file_types=[".json", ".pkl", ".pickle"]
                )
                
                load_status = gr.Textbox(
                    label="Load Status",
                    interactive=False,
                    lines=4
                )
                
                gr.Markdown("### ⚙️ Generation Settings")
                
                steering_mode = gr.Radio(
                    choices=["none", "manifold_signature"],
                    value="manifold_signature",
                    label="Steering Mode"
                )
                
                target_signature_indices = gr.Textbox(
                    label="Target Signature Indices",
                    value="0",
                    placeholder="e.g., 0,1,2",
                    info="Comma-separated indices to blend (single or multiple)"
                )
                
                steering_strength = gr.Slider(
                    minimum=0.0, maximum=3.0, value=1.0, step=0.1,
                    label="Steering Strength"
                )
                
                max_tokens = gr.Slider(
                    minimum=50, maximum=500, value=150, step=10,
                    label="Max Tokens",
                    info="Maximum length of response"
                )
                
                temperature = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.8, step=0.1,
                    label="Temperature"
                )

                # Basis selection controls
                basis_mode = gr.Radio(
                    choices=["top1", "single", "blend"],
                    value="top1",
                    label="Basis Mode (singular vectors)"
                )
                single_mode_index = gr.Slider(
                    minimum=0, maximum=7, value=0, step=1,
                    label="Single Mode Index"
                )
                blend_mode_indices = gr.Textbox(
                    label="Blend Mode Indices",
                    placeholder="e.g., 0,1,2"
                )
                blend_mode_weights = gr.Textbox(
                    label="Blend Weights (optional)",
                    placeholder="e.g., 0.6,0.4"
                )
                
                clear_btn = gr.Button("🗑️ Clear History", variant="secondary")
            
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Conversation")
                
                chatbot = gr.Chatbot(
                    height=600,
                    label="Chat History"
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your Message",
                        placeholder="Type your message here...",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
        
        # Event handlers
        manifold_file.change(
            fn=app.load_manifold,
            inputs=[manifold_file],
            outputs=[load_status]
        )
        
        # Send message on button click or Enter
        msg.submit(
            fn=app.chat,
            inputs=[msg, steering_mode, target_signature_indices, 
                   steering_strength, max_tokens, temperature, chatbot,
                   basis_mode, single_mode_index, blend_mode_indices, blend_mode_weights],
            outputs=[chatbot, msg]
        )
        
        send_btn.click(
            fn=app.chat,
            inputs=[msg, steering_mode, target_signature_indices,
                   steering_strength, max_tokens, temperature, chatbot,
                   basis_mode, single_mode_index, blend_mode_indices, blend_mode_weights],
            outputs=[chatbot, msg]
        )
        
        clear_btn.click(
            fn=app.clear_history,
            outputs=[chatbot]
        )
    
    return interface


if __name__ == "__main__":
    interface = create_chat_interface()
    interface.launch(
        server_name="127.0.0.1",
        server_port=7861,  # Different port from main interface
        share=False,
        show_error=True
    )

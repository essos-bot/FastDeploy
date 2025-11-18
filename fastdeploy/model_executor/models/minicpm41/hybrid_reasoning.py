"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
"""
Hybrid Reasoning Mode for MiniCPM4.1

This module implements hybrid reasoning mode support for MiniCPM4.1, which includes:
- Thinking token generation and processing
- Dynamic switching between reasoning and non-reasoning modes
- Context management for reasoning chains
"""

import random
from typing import Dict, List, Optional, Tuple

import paddle

from fastdeploy.model_executor.models.minicpm41.config_minicpm41 import (
    MiniCPM41HybridReasoningConfig,
)


class HybridReasoningMode:
    """
    Hybrid Reasoning Mode processor for MiniCPM4.1

    This class handles the generation and processing of thinking tokens,
    enabling the model to perform deep reasoning through structured thinking chains.
    """

    def __init__(self, config: MiniCPM41HybridReasoningConfig):
        """
        Initialize hybrid reasoning mode

        Args:
            config: Hybrid reasoning configuration
        """
        self.config = config
        self.enabled = config.enabled

        if self.enabled:
            self.reasoning_token = config.reasoning_token
            self.max_thinking_length = config.max_thinking_length
            self.thinking_probability = config.thinking_probability
            self.force_thinking = config.force_thinking

            # Track reasoning state
            self.is_thinking = False
            self.thinking_tokens_generated = 0
            self.reasoning_history = []

    def should_enter_thinking_mode(self, input_text: str = None) -> bool:
        """
        Determine whether to enter thinking mode

        Args:
            input_text: Input text to analyze (optional)

        Returns:
            bool: Whether to enter thinking mode
        """
        if not self.enabled:
            return False

        if self.force_thinking:
            return True

        if self.is_thinking:
            return True

        # Probabilistic decision based on configuration
        return random.random() < self.thinking_probability

    def process_input_with_thinking_mode(
        self,
        input_ids: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:
        """
        Process input tokens with thinking mode logic

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Tuple of (processed_input_ids, updated_attention_mask)
        """
        if not self.enabled:
            return input_ids, attention_mask

        batch_size, seq_len = input_ids.shape

        # Check if we should enter thinking mode
        if self.should_enter_thinking_mode():
            # Add thinking token to input
            thinking_token_id = self._get_thinking_token_id()
            if thinking_token_id is not None:
                # Append thinking token to sequence
                thinking_token_tensor = paddle.full(
                    [batch_size, 1],
                    thinking_token_id,
                    dtype=input_ids.dtype,
                )

                new_input_ids = paddle.concat([input_ids, thinking_token_tensor], axis=1)

                # Update attention mask if provided
                if attention_mask is not None:
                    attention_extension = paddle.ones([batch_size, 1], dtype=attention_mask.dtype)
                    new_attention_mask = paddle.concat([attention_mask, attention_extension], axis=1)
                else:
                    new_attention_mask = None

                self.is_thinking = True
                self.thinking_tokens_generated = 1

                return new_input_ids, new_attention_mask

        return input_ids, attention_mask

    def process_thinking_tokens(
        self,
        generated_token_ids: paddle.Tensor,
        logits: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, bool]:
        """
        Process generated thinking tokens

        Args:
            generated_token_ids: Generated token IDs [batch_size, 1]
            logits: Model logits [batch_size, vocab_size]

        Returns:
            Tuple of (processed_token_ids, processed_logits, should_continue_thinking)
        """
        if not self.enabled or not self.is_thinking:
            return generated_token_ids, logits, False

        batch_size = generated_token_ids.shape[0]

        # Check if we should continue thinking
        if self.thinking_tokens_generated >= self.max_thinking_length:
            self._exit_thinking_mode()
            return generated_token_ids, logits, False

        # Process thinking tokens
        self.thinking_tokens_generated += 1

        # Store in reasoning history
        for i in range(batch_size):
            token_id = generated_token_ids[i, 0].item()
            self.reasoning_history.append(
                {
                    "step": self.thinking_tokens_generated,
                    "token_id": token_id,
                    "logits": logits[i].cpu(),
                }
            )

        # Modify logits to encourage thinking-related tokens (simplified)
        # In practice, this could be more sophisticated with specific token sets
        modified_logits = self._modify_logits_for_thinking(logits)

        return generated_token_ids, modified_logits, True

    def _get_thinking_token_id(self) -> Optional[int]:
        """
        Get the token ID for the thinking token

        Returns:
            int or None: Token ID for thinking token
        """
        # This is a simplified implementation
        # In practice, you would look up the actual token ID in the tokenizer
        # For now, return a placeholder token ID
        # This should be replaced with the actual token ID lookup
        return 29971  # Placeholder token ID

    def _modify_logits_for_thinking(self, logits: paddle.Tensor) -> paddle.Tensor:
        """
        Modify logits to encourage thinking-related tokens

        Args:
            logits: Original logits [batch_size, vocab_size]

        Returns:
            Modified logits
        """
        # This is a simplified implementation
        # In practice, you would boost the probability of specific thinking-related tokens
        # For now, return logits unchanged
        return logits

    def _exit_thinking_mode(self):
        """Exit thinking mode"""
        self.is_thinking = False
        self.thinking_tokens_generated = 0

    def get_reasoning_summary(self) -> Dict:
        """
        Get a summary of the reasoning process

        Returns:
            Dict: Reasoning summary
        """
        return {
            "enabled": self.enabled,
            "is_thinking": self.is_thinking,
            "thinking_tokens_generated": self.thinking_tokens_generated,
            "max_thinking_length": self.max_thinking_length,
            "reasoning_history_length": len(self.reasoning_history),
            "reasoning_token": self.reasoning_token,
        }

    def reset(self):
        """Reset the reasoning state"""
        self.is_thinking = False
        self.thinking_tokens_generated = 0
        self.reasoning_history = []

    def should_force_non_thinking(self, input_text: str) -> bool:
        """
        Check if non-thinking mode should be forced based on input

        Args:
            input_text: Input text to analyze

        Returns:
            bool: Whether to force non-thinking mode
        """
        if not self.enabled:
            return True

        # Check for explicit non-thinking tokens in input
        # This is a simplified implementation
        non_thinking_indicators = ["/no_think", "final_answer:", "answer:"]
        input_lower = input_text.lower() if input_text else ""

        return any(indicator in input_lower for indicator in non_thinking_indicators)

    def format_input_with_thinking_mode(
        self,
        messages: List[Dict[str, str]],
        tokenizer=None,
    ) -> List[Dict[str, str]]:
        """
        Format messages with thinking mode considerations

        Args:
            messages: List of message dictionaries
            tokenizer: Tokenizer for processing (optional)

        Returns:
            List of formatted messages
        """
        if not self.enabled:
            return messages

        formatted_messages = []
        for message in messages:
            content = message.get("content", "")

            # Check for non-thinking indicators
            if self.should_force_non_thinking(content):
                # Add non-thinking indicator if not present
                if "/no_think" not in content.lower():
                    content += " /no_think"
                    formatted_messages.append({**message, "content": content})
                else:
                    formatted_messages.append(message)
            else:
                # Check if we should add thinking mode indicator
                if self.should_enter_thinking_mode(content):
                    # Add thinking token if not present
                    if self.reasoning_token not in content:
                        content += f" {self.reasoning_token}"
                        formatted_messages.append({**message, "content": content})
                    else:
                        formatted_messages.append(message)
                else:
                    formatted_messages.append(message)

        return formatted_messages


class ThinkingTokenProcessor:
    """
    Utility class for processing thinking tokens
    """

    @staticmethod
    def detect_thinking_tokens(
        token_ids: paddle.Tensor,
        thinking_token_id: int,
    ) -> paddle.Tensor:
        """
        Detect positions of thinking tokens in the sequence

        Args:
            token_ids: Token ID sequence [batch_size, seq_len]
            thinking_token_id: Token ID for thinking token

        Returns:
            Boolean tensor indicating thinking token positions
        """
        return token_ids == thinking_token_id

    @staticmethod
    def extract_thinking_content(
        token_ids: paddle.Tensor,
        thinking_token_id: int,
        tokenizer=None,
    ) -> List[str]:
        """
        Extract thinking content from token sequence

        Args:
            token_ids: Token ID sequence [batch_size, seq_len]
            thinking_token_id: Token ID for thinking token
            tokenizer: Tokenizer for decoding (optional)

        Returns:
            List of thinking content strings
        """
        thinking_positions = ThinkingTokenProcessor.detect_thinking_tokens(token_ids, thinking_token_id)

        thinking_contents = []
        for batch_idx in range(token_ids.shape[0]):
            positions = paddle.where(thinking_positions[batch_idx])[0].tolist()

            if positions:
                # Extract content after thinking tokens
                for pos in positions:
                    if pos + 1 < token_ids.shape[1]:
                        # Get tokens after thinking token
                        content_tokens = token_ids[batch_idx, pos + 1 :]
                        if tokenizer is not None:
                            content = tokenizer.decode(content_tokens.tolist())
                        else:
                            content = content_tokens.tolist()
                        thinking_contents.append(content)

        return thinking_contents

    @staticmethod
    def validate_thinking_sequence(
        token_ids: paddle.Tensor,
        thinking_token_id: int,
        max_thinking_length: int,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate thinking sequence length and structure

        Args:
            token_ids: Token ID sequence [batch_size, seq_len]
            thinking_token_id: Token ID for thinking token
            max_thinking_length: Maximum allowed thinking length

        Returns:
            Tuple of (is_valid, error_message)
        """
        thinking_positions = ThinkingTokenProcessor.detect_thinking_tokens(token_ids, thinking_token_id)

        for batch_idx in range(token_ids.shape[0]):
            positions = paddle.where(thinking_positions[batch_idx])[0].tolist()

            for pos in positions:
                # Count tokens after thinking token
                remaining_length = token_ids.shape[1] - pos - 1
                if remaining_length > max_thinking_length:
                    return False, f"Thinking sequence too long: {remaining_length} > {max_thinking_length}"

        return True, None

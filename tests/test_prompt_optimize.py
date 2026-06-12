# encoding:utf-8
"""
Unit tests for the prompt optimization feature (issue #2824).

Covers agent.chat.session_service.optimize_prompt:
  - returns the LLM-optimized text on success
  - cleans <think> tags and surrounding quotes
  - falls back to the ORIGINAL input on empty completion / exception
  - rejects empty or pathologically long results
  - includes recent context turns in the request
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestOptimizePrompt(unittest.TestCase):
    def _patched(self, reply_value):
        """Return (cm_bridge, cm_session, fake_bot) patches for a given reply."""
        fake_bot = MagicMock()
        fake_bot.reply_text.return_value = reply_value
        bridge_patch = patch("bridge.bridge.Bridge")
        session_patch = patch("models.session_manager.Session", MagicMock())
        return bridge_patch, session_patch, fake_bot

    def test_success_returns_optimized(self):
        from agent.chat.session_service import optimize_prompt
        bp, sp, bot = self._patched({"completion_tokens": 20, "content": "审查并优化代码：定位性能瓶颈。"})
        with bp as B, sp:
            B.return_value.get_bot.return_value = bot
            out = optimize_prompt("帮我看看这段代码，好像有点慢")
        self.assertEqual(out, "审查并优化代码：定位性能瓶颈。")

    def test_strips_think_tags_and_quotes(self):
        from agent.chat.session_service import optimize_prompt
        bp, sp, bot = self._patched(
            {"completion_tokens": 10, "content": '<think>reasoning</think>\n"实现用户登录接口"'}
        )
        with bp as B, sp:
            B.return_value.get_bot.return_value = bot
            out = optimize_prompt("写个登录接口")
        self.assertEqual(out, "实现用户登录接口")

    def test_empty_completion_falls_back(self):
        from agent.chat.session_service import optimize_prompt
        bp, sp, bot = self._patched({"completion_tokens": 0, "content": "我现在有点累了"})
        with bp as B, sp:
            B.return_value.get_bot.return_value = bot
            out = optimize_prompt("随便说点啥")
        self.assertEqual(out, "随便说点啥")

    def test_exception_falls_back(self):
        from agent.chat.session_service import optimize_prompt
        with patch("bridge.bridge.Bridge", side_effect=RuntimeError("boom")):
            out = optimize_prompt("hello there")
        self.assertEqual(out, "hello there")

    def test_empty_input_returns_empty(self):
        from agent.chat.session_service import optimize_prompt
        self.assertEqual(optimize_prompt("   "), "")
        self.assertEqual(optimize_prompt(None), "")

    def test_pathologically_long_result_rejected(self):
        from agent.chat.session_service import optimize_prompt
        huge = "x" * 5000
        bp, sp, bot = self._patched({"completion_tokens": 50, "content": huge})
        with bp as B, sp:
            B.return_value.get_bot.return_value = bot
            out = optimize_prompt("short input")
        # Result is way longer than max(2000, 8*len) -> fall back to original.
        self.assertEqual(out, "short input")

    def test_empty_optimized_result_rejected(self):
        from agent.chat.session_service import optimize_prompt
        bp, sp, bot = self._patched({"completion_tokens": 5, "content": '"" '})
        with bp as B, sp:
            B.return_value.get_bot.return_value = bot
            out = optimize_prompt("keep me")
        self.assertEqual(out, "keep me")

    def test_context_messages_included(self):
        from agent.chat.session_service import optimize_prompt
        bot = MagicMock()
        bot.reply_text.return_value = {"completion_tokens": 5, "content": "optimized"}
        ctx = [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
            {"role": "system", "content": "should be ignored"},
        ]
        with patch("bridge.bridge.Bridge") as B, \
             patch("models.session_manager.Session") as Sess:
            B.return_value.get_bot.return_value = bot
            sess_inst = MagicMock()
            Sess.return_value = sess_inst
            optimize_prompt("now do this", ctx)
            # The handler assigns the assembled messages onto session.messages.
            assigned = sess_inst.messages
            roles = [m["role"] for m in assigned]
            # system prompt + user/assistant context + final user input
            self.assertEqual(roles[0], "system")
            self.assertIn("user", roles)
            self.assertIn("assistant", roles)
            self.assertEqual(assigned[-1]["role"], "user")
            self.assertIn("now do this", assigned[-1]["content"])
            # 'system' context role must not be copied as a context turn
            # (only the leading optimize system prompt is system).
            self.assertEqual(roles.count("system"), 1)


if __name__ == "__main__":
    unittest.main()

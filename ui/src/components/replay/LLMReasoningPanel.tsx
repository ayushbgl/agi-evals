/**
 * LLM Reasoning Panel - Display Chain-of-Thought reasoning
 *
 * Shows LLM decision details including:
 * - Model name, latency, token usage
 * - Action taken with rationale
 * - Full CoT reasoning (collapsible)
 */

import React, { useState } from "react";
import {
  Box,
  Paper,
  Typography,
  Chip,
  Collapse,
  IconButton,
  Divider,
  Tooltip,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import SmartToyIcon from "@mui/icons-material/SmartToy";
import TimerIcon from "@mui/icons-material/Timer";
import TokenIcon from "@mui/icons-material/DataUsage";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ErrorIcon from "@mui/icons-material/Error";

import type { LLMDecision, GameAction } from "../../utils/arenaApi.types";

interface LLMReasoningPanelProps {
  decision: LLMDecision | null | undefined;
  action: GameAction;
  playerColor: string;
  showReasoning: boolean;
}

const COLOR_MAP: Record<string, string> = {
  RED: "#e53935",
  BLUE: "#1e88e5",
  ORANGE: "#fb8c00",
  WHITE: "#9e9e9e",
};

export function LLMReasoningPanel({
  decision,
  action,
  playerColor,
  showReasoning,
}: LLMReasoningPanelProps) {
  const [expanded, setExpanded] = useState(false);

  // Format action for display
  const formatAction = (act: GameAction): string => {
    let str = act.action_type;
    if (act.value !== undefined && act.value !== null) {
      if (typeof act.value === "object") {
        str += `: ${JSON.stringify(act.value)}`;
      } else {
        str += `: ${act.value}`;
      }
    }
    return str;
  };

  // Format token count
  const formatTokens = (tokens: number): string => {
    if (tokens >= 1000) {
      return `${(tokens / 1000).toFixed(1)}k`;
    }
    return tokens.toString();
  };

  const borderColor = COLOR_MAP[playerColor] || "#888";

  return (
    <Paper
      elevation={2}
      sx={{
        p: 2,
        borderLeft: `4px solid ${borderColor}`,
        backgroundColor: "background.paper",
      }}
    >
      {/* Header */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          mb: 1,
        }}
      >
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <SmartToyIcon fontSize="small" sx={{ color: borderColor }} />
          <Typography variant="subtitle1" sx={{ fontWeight: "bold" }}>
            {decision?.model_name || "Unknown Model"}
          </Typography>
          <Chip
            label={playerColor}
            size="small"
            sx={{
              backgroundColor: borderColor,
              color: playerColor === "WHITE" ? "#000" : "#fff",
            }}
          />
        </Box>

        {decision && (
          <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
            <Tooltip title="Response latency">
              <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                <TimerIcon fontSize="small" color="action" />
                <Typography variant="body2" color="text.secondary">
                  {decision.latency_ms}ms
                </Typography>
              </Box>
            </Tooltip>

            <Tooltip title="Total tokens used">
              <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                <TokenIcon fontSize="small" color="action" />
                <Typography variant="body2" color="text.secondary">
                  {formatTokens(decision.total_tokens)}
                </Typography>
              </Box>
            </Tooltip>

            {decision.parsed_successfully ? (
              <Tooltip title="Action parsed successfully">
                <CheckCircleIcon fontSize="small" color="success" />
              </Tooltip>
            ) : (
              <Tooltip title={`Parse error: ${decision.parse_error || "Unknown"}`}>
                <ErrorIcon fontSize="small" color="error" />
              </Tooltip>
            )}
          </Box>
        )}
      </Box>

      <Divider sx={{ my: 1 }} />

      {/* Action taken */}
      <Box sx={{ mb: 1 }}>
        <Typography variant="body2" color="text.secondary">
          Action:
        </Typography>
        <Typography
          variant="body1"
          sx={{
            fontFamily: "monospace",
            backgroundColor: "action.hover",
            p: 1,
            borderRadius: 1,
            mt: 0.5,
          }}
        >
          {formatAction(action)}
        </Typography>
        {action.rationale && (
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{ mt: 0.5, fontStyle: "italic" }}
          >
            "{action.rationale}"
          </Typography>
        )}
      </Box>

      {/* Reasoning section - only show if allowed by view mode */}
      {showReasoning && decision?.reasoning && (
        <>
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              cursor: "pointer",
              mt: 1,
            }}
            onClick={() => setExpanded(!expanded)}
          >
            <Typography variant="body2" color="text.secondary">
              Chain-of-Thought Reasoning
            </Typography>
            <IconButton size="small">
              {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            </IconButton>
          </Box>

          <Collapse in={expanded}>
            <Paper
              variant="outlined"
              sx={{
                p: 1.5,
                mt: 1,
                backgroundColor: "grey.50",
                maxHeight: 300,
                overflow: "auto",
              }}
            >
              <Typography
                variant="body2"
                sx={{
                  whiteSpace: "pre-wrap",
                  fontFamily: "monospace",
                  fontSize: "0.85rem",
                }}
              >
                {decision.reasoning}
              </Typography>
            </Paper>
          </Collapse>
        </>
      )}

      {/* Token breakdown (if expanded and available) */}
      {expanded && decision && (
        <Box sx={{ mt: 1, display: "flex", gap: 2 }}>
          <Typography variant="caption" color="text.secondary">
            Prompt: {formatTokens(decision.prompt_tokens)} tokens
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Completion: {formatTokens(decision.completion_tokens)} tokens
          </Typography>
        </Box>
      )}

      {/* No decision available */}
      {!decision && (
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          No LLM decision data (player may be random/minimax)
        </Typography>
      )}
    </Paper>
  );
}

export default LLMReasoningPanel;

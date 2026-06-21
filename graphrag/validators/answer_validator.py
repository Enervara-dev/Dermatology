from __future__ import annotations

import json
import logging
from typing import Generator, Optional

from pydantic import TypeAdapter, ValidationError
from graphrag.schemas.blocks import Block

logger = logging.getLogger(__name__)

_block_adapter = TypeAdapter(Block)

def validate_line(line: str) -> Block | None:
    """
    Validate a single line as a single Block.
    On failure, log the reason and return None (drops the line).
    """
    line = line.strip()
    if not line:
        return None
    try:
        data = json.loads(line)
        block = _block_adapter.validate_python(data)
        return block
    except json.JSONDecodeError as e:
        logger.warning(f"Line is not valid JSON: {e}. Line: {line}")
        return None
    except ValidationError as e:
        logger.warning(f"Block validation error: {e}. Data: {line}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected validation error: {e}. Line: {line}")
        return None

def iter_blocks(token_stream, *, terminal: bool) -> Generator[Block, None, None]:
    """
    Line-buffer a token stream, validate each line, drop follow_up_questions when
    terminal is True, and flush the trailing line.
    """
    buffer = ""
    for token in token_stream:
        buffer += token
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            block = validate_line(line)
            if block is not None:
                if terminal and block.type == "follow_up_questions":
                    logger.info("Dropping follow-up questions block under terminal constraint.")
                    continue
                yield block

    # Flush any remaining content in the buffer (the model may omit the final \n)
    if buffer:
        block = validate_line(buffer)
        if block is not None:
            if not (terminal and block.type == "follow_up_questions"):
                yield block

import json
import logging

import pika

from src.config import config

logger = logging.getLogger(__name__)


def publish(channel, payload: dict) -> None:
    """Publish a result payload to the AMQP result queue."""
    channel.queue_declare(queue=config.AMQP_RESULT_QUEUE, durable=True)
    channel.basic_publish(
        exchange="",
        routing_key=config.AMQP_RESULT_QUEUE,
        body=json.dumps(payload),
        properties=pika.BasicProperties(
            delivery_mode=2,        # persistent
            content_type="application/json",
        ),
    )
    logger.info("Published result to '%s'", config.AMQP_RESULT_QUEUE)

"""
AMQP consumer — listens on `ats_check_queue`, scores resumes, prints to console, publishes to `ats_check_result`.

Run with:
    python -m worker.consumer
"""

import json
import logging
import sys

import pika

from src.config import config
from worker import handler, publisher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def on_message(ch, method, _properties, body):
    job_id = "(unknown)"
    try:
        payload = json.loads(body)
        job_id = payload.get("id") or payload.get("job_id") or job_id

        handler.validate(payload)
        result = handler.process(payload)
        handler.print_result(job_id, result)

        publisher.publish(ch, handler.build_payload(job_id, result))

        ch.basic_ack(delivery_tag=method.delivery_tag)

    except (ValueError, KeyError) as exc:
        logger.error("[%s] Validation/processing error: %s", job_id, exc)
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    except Exception as exc:
        logger.exception("[%s] Unexpected error: %s", job_id, exc)
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def start():
    credentials = pika.PlainCredentials(config.AMQP_USERNAME, config.AMQP_PASSWORD)
    params = pika.ConnectionParameters(
        host=_parse_host(config.AMQP_URL),
        port=_parse_port(config.AMQP_URL),
        credentials=credentials,
        heartbeat=60,
        blocked_connection_timeout=30,
    )

    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    channel.queue_declare(queue=config.AMQP_QUEUE, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=config.AMQP_QUEUE, on_message_callback=on_message)

    logger.info("Waiting for messages on queue '%s'. CTRL+C to exit.", config.AMQP_QUEUE)
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        channel.stop_consuming()
    finally:
        connection.close()


def _parse_host(url: str) -> str:
    # amqp://host:port  →  host
    return url.split("//")[-1].split(":")[0]


def _parse_port(url: str) -> int:
    # amqp://host:port  →  port (default 5672)
    try:
        return int(url.split(":")[-1])
    except ValueError:
        return 5672


if __name__ == "__main__":
    start()

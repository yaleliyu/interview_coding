# Design a rate limiter class in Python.
# It limits how often an action can be performed.
# The core method is something like allow(key) -> bool:
#   it returns True if the action for that key is permitted right now,
#   False if the caller has exceeded their allowed rate.
#   Different keys (e.g., per user, per IP) are limited independently.

import threading
import time


class RateLimiter:
    def __init__(self, limit_req, limit_window):
        self._tokens = dict()
        self._lock_global = threading.RLock()


        self._limit_req = limit_req
        self._limit_window = limit_window
        self._refill_rate = self._limit_req * 1.0 / self._limit_window


    def allow(self, key):
        time_stamp = time.monotonic()

        with self._lock_global:
            if key not in self._tokens:
                local_token = dict()
                local_token['last_time'] = time_stamp
                local_token['last_capacity'] = self._limit_req
                local_token['lock'] = threading.RLock()
                self._tokens[key] = local_token

            token = self._tokens[key]

        with token['lock']:
            capacity = min(self._limit_req, token['last_capacity'] + (time_stamp- token['last_time']) * self._refill_rate)

            if capacity >= 1:
                token['last_time'] = time_stamp
                token['last_capacity'] = capacity - 1
                return True

        return False

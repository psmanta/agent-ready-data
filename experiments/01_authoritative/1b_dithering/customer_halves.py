#!/usr/bin/env python3
"""
Customer halves: the replication split (Amendment 1, Revision 3, R3-4).
=====================================================================
half(c) is the parity of the first byte of SHA-256 of "adc-1b-halves-v1|<customer_id>". Halves are by CUSTOMER, never by
block, so every record of a customer (baseline runs, every condition, the control) sits in the same half and paired
comparisons stay paired. Deterministic, independent of n and of the order customers are listed in. On the 1,000 ids
CUST_000001..CUST_001000 the split is 486 (half 0) and 514 (half 1). The string is a namespace, not a secret; a new
string needs a new revision of the amendment.

    python3 customer_halves.py            # prints the split for CUST_000001..CUST_001000
"""
import hashlib
from typing import Dict, Iterable, List

PARTITION = "adc-1b-halves-v1"


def half(customer_id: str) -> int:
    return hashlib.sha256(f"{PARTITION}|{customer_id}".encode()).digest()[0] % 2


def split(customer_ids: Iterable[str]) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {0: [], 1: []}
    for c in customer_ids:
        out[half(c)].append(c)
    return out


if __name__ == "__main__":
    s = split(f"CUST_{i:06d}" for i in range(1, 1001))
    print(f"{PARTITION}: half 0 = {len(s[0])}, half 1 = {len(s[1])}")

"""Deterministic text features for the DS 6021 tweet-pattern model."""

import csv
import html
import io
import re
import unicodedata
from itertools import groupby

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

CSV_ROOT = "Twibot-20 Data"
TWEET_LIMIT = 24
MIN_TWEETS = 8
SCAN_LIMIT = 96
RETWEET = re.compile(r"^RT\s+@\w+\s*:?\s*", re.IGNORECASE)
URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HANDLE = re.compile(r"(?<!\w)@\w+")
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
NUMBER = re.compile(r"\d+")
REPEATED_PUNCTUATION = re.compile(r"[!?]{2,}")
# An approximation over common emoji blocks, measured in code points rather than graphemes.
EMOJI_SYMBOL = re.compile(
    r"[\u2600-\u26ff\u2700-\u27bf\U0001f1e6-\U0001f1ff"
    r"\U0001f300-\U0001f6ff\U0001f7e0-\U0001f7eb\U0001f900-\U0001faff]"
)
PERSONAL = frozenset("i me my mine myself we us our ours you your yours".split())
NAMES = (
    "retweet_rate", "link_rate", "mention_prefix_rate", "hashtag_rate",
    "template_reuse", "exact_template_rate", "word_diversity",
    "mean_length", "length_cv", "punctuation_rate",
    "personal_word_rate", "question_rate", "function_word_rate", "mean_word_length",
)
STYLE_NAMES = (
    "emoji_post_rate", "emoji_codepoint_rate", "em_dash_post_rate", "exclamation_post_rate",
    "ellipsis_post_rate", "repeated_punctuation_post_rate", "uppercase_letter_rate",
    "uppercase_word_rate", "title_case_post_rate",
)
ACTIVITY_PROXY_NAMES = (
    "link_retweet_post_rate", "broadcast_switch_rate", "longest_broadcast_run_share",
)
EXTENDED_NAMES = NAMES + STYLE_NAMES + ACTIVITY_PROXY_NAMES


def csv_rows(archive, filename, fields):
    with archive.open(f"{CSV_ROOT}/{filename}") as entry:
        reader = csv.DictReader(io.TextIOWrapper(entry, encoding="utf-8-sig", newline=""))
        if not set(fields).issubset(reader.fieldnames or []):
            raise ValueError(f"Missing required fields in {filename}.")
        yield from reader


def style_features(bodies):
    emoji_counts = [
        sum(unicodedata.category(symbol) in {"So", "Sk"} for symbol in EMOJI_SYMBOL.findall(body))
        for body in bodies
    ]
    words_by_post = [[word for word in WORD.findall(body) if len(word) > 1] for body in bodies]
    words = [word for post_words in words_by_post for word in post_words]
    uppercase = sum(char.isupper() for body in bodies for char in body)
    lowercase = sum(char.islower() for body in bodies for char in body)
    return (
        np.mean([count > 0 for count in emoji_counts]),
        sum(emoji_counts) / max(1, sum(map(len, bodies))),
        np.mean(["\u2014" in body for body in bodies]),
        np.mean(["!" in body for body in bodies]),
        np.mean(["..." in body for body in bodies]),
        np.mean([bool(REPEATED_PUNCTUATION.search(body)) for body in bodies]),
        uppercase / max(1, uppercase + lowercase),
        sum(word.isupper() for word in words) / max(1, len(words)),
        np.mean([
            len(post_words) >= 4 and sum(word.istitle() for word in post_words) >= 0.6 * len(post_words)
            for post_words in words_by_post
        ]),
    )


def activity_proxies(flags):
    # Eligibility-filtered tweet index order describes transitions, not elapsed time.
    broadcasts = [retweet or link for retweet, link, _, _ in flags]
    longest_run = max((sum(1 for _ in run) for broadcast, run in groupby(broadcasts) if broadcast), default=0)
    return (
        np.mean([retweet and link for retweet, link, _, _ in flags]),
        np.mean([left != right for left, right in zip(broadcasts, broadcasts[1:])]),
        longest_run / len(broadcasts),
    )


def summarize(timeline, *, extended=False):
    texts, words_by_tweet, flags, case_preserved = [], [], [], []
    for _, raw in sorted(timeline.items())[:SCAN_LIMIT]:
        normalized = unicodedata.normalize("NFKC", html.unescape(raw)).strip()
        raw = normalized.casefold()
        is_retweet = bool(RETWEET.match(raw))
        body = " ".join(HANDLE.sub(" ", URL.sub(" ", RETWEET.sub("", raw))).split())
        words = WORD.findall(body)
        if len(words) < 3 or sum(word in ENGLISH_STOP_WORDS for word in words) < 2:
            continue
        if sum(word.isascii() for word in words) < 0.85 * len(words):
            continue
        texts.append(body)
        words_by_tweet.append(words)
        flags.append((is_retweet, bool(URL.search(raw)), raw.startswith("@"), "#" in raw))
        if extended:
            visible = RETWEET.sub("", normalized)
            case_preserved.append(" ".join(HANDLE.sub(" ", URL.sub(" ", visible)).split()))
        if len(texts) == TWEET_LIMIT:
            break

    if len(texts) < MIN_TWEETS:
        return np.full(len(EXTENDED_NAMES if extended else NAMES), np.nan)

    templates = [" ".join(WORD.findall(NUMBER.sub(" number ", text))) for text in texts]
    grams = [{text[i:i + 4] for i in range(len(text) - 3)} for text in templates]
    nearest = np.zeros(len(grams))
    for index, left in enumerate(grams):
        for other in range(index + 1, len(grams)):
            right = grams[other]
            score = len(left & right) / len(left | right) if left or right else 0.0
            nearest[index] = max(nearest[index], score)
            nearest[other] = max(nearest[other], score)

    words = [word for group in words_by_tweet for word in group]
    lengths = np.array([len(text) for text in texts], dtype=float)
    base_features = np.array([
        *np.mean(flags, axis=0),
        nearest.mean(),
        1 - len(set(templates)) / len(templates),
        len(set(words)) / len(words),
        lengths.mean(),
        lengths.std() / lengths.mean(),
        sum(sum(char in "!?,.;:" for char in text) for text in texts) / lengths.sum(),
        sum(word in PERSONAL for word in words) / len(words),
        sum("?" in text for text in texts) / len(texts),
        sum(word in ENGLISH_STOP_WORDS for word in words) / len(words),
        sum(map(len, words)) / len(words),
    ])
    if extended:
        return np.concatenate((base_features, style_features(case_preserved), activity_proxies(flags)))
    return base_features


def timelines(rows):
    seen = set()
    for account, group in groupby(rows, lambda row: (row["ID"] or "").strip()):
        if not account or account in seen:
            raise ValueError("Tweets must have a nonblank, contiguous account ID.")
        seen.add(account)
        timeline = {}
        for row in group:
            index = int(row["tweet_idx"])
            if index < 0 or index in timeline:
                raise ValueError("Tweets must have unique, nonnegative indices per account.")
            timeline[index] = row["text"] or ""
        yield account, timeline


def extract(archive, ids, split="train", *, extended=False):
    rows = csv_rows(archive, f"{split}_tweets.csv", ("ID", "tweet_idx", "text"))
    expected = set(ids)
    values = {}
    for account, timeline in timelines(rows):
        if account not in expected:
            raise ValueError(f"{split} tweet account ID is absent from the profile table.")
        values[account] = summarize(timeline, extended=extended)
    missing = np.full(len(EXTENDED_NAMES if extended else NAMES), np.nan)
    return np.vstack([values.get(account, missing) for account in ids])

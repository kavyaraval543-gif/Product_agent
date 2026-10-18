"""robots.txt handling. Fixtures are real excerpts fetched on 2026-10-17."""

from pda.robots import RobotsRules, parse_robots

TOKEN = "ProductDiscoveryAgent"

# Verbatim from https://itunes.apple.com/robots.txt (the reason Apple's review RSS was rejected).
APPLE = """User-agent: *
Disallow: /WebObjects/MZFastFinance.woa
Disallow: /WebObjects/*
Allow: /WebObjects/MZStore.woa/wa/viewMultiRoom?*
Disallow: /search*
Disallow: /*/rss/*
Disallow: /*/lookup?

User-agent: Googlebot
Disallow: /*/album/*/*?i=*
"""

# Verbatim excerpt from https://addons.mozilla.org/robots.txt
AMO = """# robots.txt file for Mozilla Add-ons, addons.mozilla.org
User-agent: *
Allow: /

Disallow: /firefox/downloads/
Disallow: /android/downloads/
Allow: /cs/firefox/collections/mozilla/
Disallow: /cs/firefox/collections/mozilla/$
Disallow: /cs/firefox/search/
"""


def test_apple_review_rss_is_disallowed_by_wildcard_rule():
    rules = parse_robots(APPLE, TOKEN)
    allowed, rule = rules.decide("/us/rss/customerreviews/page=1/id=570060128/sortby=mostrecent/json")
    assert not allowed
    assert rule is not None and rule.pattern == "/*/rss/*"


def test_apple_lookup_and_search_rules():
    rules = parse_robots(APPLE, TOKEN)
    assert not rules.is_allowed("/us/lookup?id=570060128")  # matches /*/lookup?
    assert rules.is_allowed("/lookup?id=570060128")  # no leading country segment: rule does not match
    assert not rules.is_allowed("/search?term=duolingo")  # /search*


def test_amo_ratings_api_is_allowed_and_downloads_are_not():
    rules = parse_robots(AMO, TOKEN)
    assert rules.is_allowed("/api/v5/ratings/rating/?addon=bitwarden-password-manager&page_size=50")
    assert not rules.is_allowed("/firefox/downloads/file/123/x.xpi")
    assert not rules.is_allowed("/cs/firefox/search/?q=x")


def test_longest_match_wins_and_allow_wins_ties():
    rules = parse_robots("User-agent: *\nDisallow: /\nAllow: /public/\n", TOKEN)
    assert rules.is_allowed("/public/page")
    assert not rules.is_allowed("/private")
    tie = parse_robots("User-agent: *\nDisallow: /same\nAllow: /same\n", TOKEN)
    assert tie.is_allowed("/same")


def test_dollar_anchor_and_star():
    rules = parse_robots("User-agent: *\nDisallow: /*.json$\n", TOKEN)
    assert not rules.is_allowed("/a/b.json")
    assert rules.is_allowed("/a/b.json?x=1")  # '$' means end of URL, so a query string escapes the rule
    assert rules.is_allowed("/a/b.jsonp")


def test_specific_group_overrides_wildcard_group():
    text = "User-agent: *\nDisallow: /\n\nUser-agent: ProductDiscoveryAgent\nAllow: /\nDisallow: /secret\n"
    rules = parse_robots(text, TOKEN)
    assert rules.is_allowed("/anything")
    assert not rules.is_allowed("/secret")


def test_other_bots_group_does_not_apply_to_us():
    rules = parse_robots("User-agent: Googlebot\nDisallow: /\n", TOKEN)
    assert rules.is_allowed("/anything")


def test_empty_disallow_and_empty_file_allow_everything():
    assert parse_robots("User-agent: *\nDisallow:\n", TOKEN).is_allowed("/x")
    assert parse_robots("", TOKEN).is_allowed("/x")


def test_multiple_user_agent_lines_share_a_group():
    rules = parse_robots("User-agent: a\nUser-agent: *\nDisallow: /x\n", TOKEN)
    assert not rules.is_allowed("/x")


def test_comments_and_case_insensitivity():
    rules = parse_robots("USER-AGENT: *   # everyone\ndisallow: /nope  # no\n", TOKEN)
    assert not rules.is_allowed("/nope")


def test_forbidding_and_permissive_constants():
    assert not RobotsRules.forbidding().is_allowed("/")
    assert RobotsRules.permissive().is_allowed("/anything")

import asyncio

from mcp_servers.home.player import Player


def _player(tmp_path):
    for name in ("A - one.ogg", "B - two.ogg", "C - three.ogg"):
        (tmp_path / name).write_bytes(b"x")
    return Player(tmp_path, tmp_path / "mpv.sock", browser=True)


def test_browser_mode_never_starts_mpv_and_reports_the_file(tmp_path):
    p = _player(tmp_path)
    assert p.simulated  # the laptop plays the audio, not the Pi
    assert asyncio.run(p.play()) == "Playing A - one."
    info = p.info()
    assert info["music_playing"] and not info["music_stopped"] and info["music_browser"]
    assert info["music_file"] == "A - one.ogg"


def test_next_end_of_track_pause_stop_volume(tmp_path):
    p = _player(tmp_path)
    asyncio.run(p.play())
    assert asyncio.run(p.next()) == "Skipping to B - two." and p.info()["music_playing"]
    p.track_ended()                                   # browser finished B -> C, still playing
    assert p.info()["music_file"] == "C - three.ogg" and p.info()["music_playing"]
    p.track_ended()                                   # wraps around
    assert p.info()["music_file"] == "A - one.ogg"
    asyncio.run(p.pause())
    assert not p.info()["music_playing"] and not p.info()["music_stopped"]
    asyncio.run(p.stop())
    assert p.info()["music_stopped"]
    assert asyncio.run(p.change_volume(10)) == "Volume at 70 percent." and p.info()["volume"] == 70

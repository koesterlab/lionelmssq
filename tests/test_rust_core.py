def test_rust_hello():
    from spectrseqtools.rust_core import hello

    assert "Hello, world!" == hello("world")

# Releasing

A version is published by pushing its tag: `.github/workflows/release.yml`
then builds the package, publishes it on PyPI, pushes the image to ghcr.io and
creates the GitHub release with the changelog's entry.

## Once, before the first release

1. On [pypi.org](https://pypi.org), with the maintainer's account: Your
   projects, Publishing, "Add a new pending publisher": PyPI project name
   `whenframe`, owner `qthullie`, repository `whenframe`, workflow
   `release.yml`, environment `pypi`. PyPI then trusts that workflow for that
   project, and no token is stored anywhere.
2. On GitHub: Settings, Environments, New environment `pypi`. Requiring a
   reviewer there makes every release wait for a click.
3. After the first release: the package `whenframe` on the GitHub profile,
   Package settings, visibility public, so that `docker pull` works without
   logging in.

## Each release

1. The version in `whenframe/__init__.py` (the package reads it from there).
2. Its entry in `CHANGELOG.md`, headed `## 0.1.0` with the version alone: the
   workflow refuses a tag without one.
3. `make check`, then commit, then:

```bash
git tag v0.1.0
git push origin v0.1.0
```

A version cannot be published twice on PyPI: a mistake found after the push
is fixed by the next version.

After the first release, the install lines of the README and of
docs/TUTORIAL.md can become `pip install whenframe` (with `[speech]` as
before) and `docker pull ghcr.io/qthullie/whenframe`.

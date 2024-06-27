"""Automation using nox"""

import nox

nox.options.reuse_existing_virtualenvs = True

# AUTHORS_FILE = "AUTHORS.txt"
VERSION_FILE = "src/pip/__init__.py"


@nox.session
def tests(session):
    """Run tests via Nox"""
    session.install('-r', '../requirements.txt')
    session.run('pytest')


@nox.session
def lint(session):
    """Run linter via Nox"""
    session.install('-r', '../requirements.txt')
    session.run('flake8')
    session.run('pylint')

#!/usr/bin/env python

## @package project
#  This package will call neo4j and es
#  so that it can start the django server.
#
#  More details.

"""Django's command-line utility for administrative tasks."""
import os
import subprocess
import sys
from scripts.esScript import close_es,open_es
from scripts.neoScript import close_neo4j,open_neo4j
## Documentation for a function.
#
#  More details.
def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc

    open_es()
    open_neo4j()
    execute_from_command_line(sys.argv)
    close_es()
    close_neo4j()


if __name__ == '__main__':
    main()

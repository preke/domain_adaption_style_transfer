import logging

logging_config = dict(
    version=1,
    formatters={
        'simple': {'format': '%(asctime)s %(levelname)s {path: %(pathname)s Line_No: %(lineno)d} %(message)s'}
    },
    handlers={
        'default_handlers': {'class': 'logging.handlers.RotatingFileHandler',
                             'filename': './logfile/logger.log',
                             'maxBytes': 1024 * 1024 * 20,
                             'backupCount': 50,
                             'level': 'WARNING',
                             'formatter': 'simple',
                             'encoding': 'utf8'}
    },

    root={
        'handlers': ['default_handlers'],
        'level': logging.WARNING,
    },
)

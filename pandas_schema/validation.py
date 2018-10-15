from __future__ import absolute_import
import abc
import math
import datetime
import pandas as pd
import numpy as np
import typing
import operator

from .validation_warning import ValidationWarning
from .errors import PanSchArgumentError


class _BaseValidation(object):
    u"""
    The validation base class that defines any object that can create a list of errors from a Series
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_errors(self, series, column):
        u"""
        Return a list of errors in the given series
        :param series:
        :param column:
        :return:
        """


class SeriesValidation(_BaseValidation):
    u"""
    Implements the _BaseValidation interface by returning a Boolean series for each element that either passes or
    fails the validation
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        self._custom_message = kwargs.get(u'message')

    @property
    def message(self):
        return self._custom_message or self.default_message

    @abc.abstractproperty
    def default_message(self):
        u"""
        Create a message to be displayed whenever this validation fails
        This should be a generic message for the validation type, but can be overwritten if the user provides a
        message kwarg
        """

    @abc.abstractmethod
    def validate(self, series):
        u"""
        Returns a Boolean series, where each value of False is an element in the Series that has failed the validation
        :param series:
        :return:
        """

    def __invert__(self):
        u"""
        Returns a negated version of this validation
        """
        return InverseValidation(self)

    def __or__(self, other):
        u"""
        Returns a validation which is true if either this or the other validation is true
        """
        return _CombinedValidation(self, other, operator.or_)

    def __and__(self, other):
        u"""
        Returns a validation which is true if either this or the other validation is true
        """
        return _CombinedValidation(self, other, operator.and_)

    def get_errors(self, series, column):

        errors = []

        # Calculate which columns are valid using the child class's validate function, skipping empty entries if the
        # column specifies to do so
        simple_validation = ~self.validate(series)
        if column.allow_empty:
            # Failing results are those that are not empty, and fail the validation
            validated = (series.unicode.len() > 0) & simple_validation
        else:
            validated = simple_validation

        # Cut down the original series to only ones that failed the validation
        indices = series.index[validated]

        # Use these indices to find the failing items. Also print the index which is probably a row number
        for i in indices:
            element = series[i]
            errors.append(ValidationWarning(
                message=self.message,
                value=element,
                row=i,
                column=series.name
            ))

        return errors


class InverseValidation(SeriesValidation):
    u"""
    Negates an ElementValidation. Automatically created with ~validation, but can be used manually to specify a custom
    message
    """

    def __init__(self, validation, **kwargs):
        self.negated = validation
        super(InverseValidation, self).__init__(**kwargs)

    def validate(self, series):
        return ~ self.negated.validate(series)

    @property
    def default_message(self):
        return self.negated.message + u' <negated>'


class _CombinedValidation(SeriesValidation):
    u"""
    Validates if one and/or the other validation is true for an element
    """

    def __init__(self, validation_a, validation_b, operator):
        self.operator = operator
        self.v_a = validation_a
        self.v_b = validation_b
        super(_CombinedValidation, self).__init__()

    def validate(self, series):
        return self.operator(self.v_a.validate(series), self.v_b.validate(series))

    @property
    def default_message(self):
        return u'({}) {} ({})'.format(self.v_a.message, self.operator, self.v_b.message)


class CustomSeriesValidation(SeriesValidation):
    u"""
    Validates using a user-provided function that operates on an entire series (for example by using one of the pandas
    Series methods: http://pandas.pydata.org/pandas-docs/stable/api.html#series)
    """

    def __init__(self, validation, message):
        u"""
        :param message: The error message to provide to the user if this validation fails. The row and column and
            failing value will automatically be prepended to this message, so you only have to provide a message that
            describes what went wrong, for example 'failed my validation' will become

            {row: 1, column: "Column Name"}: "Value" failed my validation
        :param validation: A function that takes a pandas Series and returns a boolean Series, where each cell is equal
            to True if the object passed validation, and False if it failed
        """
        self._validation = validation
        super(CustomSeriesValidation, self).__init__(message=message)

    def validate(self, series):
        return self._validation(series)

    @property
    def default_message(self):
        return self.negated.message + u' <negated>'


class CustomElementValidation(SeriesValidation):
    u"""
    Validates using a user-provided function that operates on each element
    """

    def __init__(self, validation, message):
        u"""
        :param message: The error message to provide to the user if this validation fails. The row and column and
            failing value will automatically be prepended to this message, so you only have to provide a message that
            describes what went wrong, for example 'failed my validation' will become

            {row: 1, column: "Column Name"}: "Value" failed my validation
        :param validation: A function that takes the value of a data frame cell and returns True if it passes the
            the validation, and false if it doesn't
        """
        self._validation = validation
        super(CustomElementValidation, self).__init__(message=message)

    def validate(self, series):
        return series.apply(self._validation)


class InRangeValidation(SeriesValidation):
    u"""
    Checks that each element in the series is within a given numerical range
    """

    def __init__(self, min = -float('inf'), max = float('inf'), **kwargs):
        u"""
        :param min: The minimum (inclusive) value to accept
        :param max: The maximum (exclusive) value to accept
        """
        self.min = min
        self.max = max
        super(InRangeValidation, self).__init__(**kwargs)

    @property
    def default_message(self):
        return u'was not in the range [{}, {})'.format(self.min, self.max)

    def validate(self, series):
        series = pd.to_numeric(series)
        return (series >= self.min) & (series < self.max)


class IsDtypeValidation(_BaseValidation):
    u"""
    Checks that a series has a certain numpy dtype
    """

    def __init__(self, dtype, **kwargs):
        u"""
        :param dtype: The numpy dtype to check the column against
        """
        self.dtype = dtype
        super(IsDtypeValidation, self).__init__(**kwargs)

    def get_errors(self, series, column = None):
        if not np.issubdtype(series.dtype, self.dtype):
            return [ValidationWarning(
                u'The column has a dtype of {} which is not a subclass of the required type {}'.format(series.dtype,
                                                                                                      self.dtype))]
        else:
            return []


class CanCallValidation(SeriesValidation):
    u"""
    Validates if a given function can be called on each element in a column without raising an exception
    """

    def __init__(self, func, **kwargs):
        u"""
        :param func: A python function that will be called with the value of each cell in the DataFrame. If this
            function throws an error, this cell is considered to have failed the validation. Otherwise it has passed.
        """
        if callable(type):
            self.callable = func
        else:
            raise PanSchArgumentError(u'The object "{}" passed to CanCallValidation is not callable!'.format(type))
        super(CanCallValidation, self).__init__(**kwargs)

    @property
    def default_message(self):
        return u'raised an exception when the callable {} was called on it'.format(self.callable)

    def can_call(self, var):
        try:
            self.callable(var)
            return True
        except:
            return False

    def validate(self, series):
        return series.apply(self.can_call)


class CanConvertValidation(CanCallValidation):
    u"""
    Checks if each element in a column can be converted to a Python object type
    """

    u"""
    Internally this uses the same logic as CanCallValidation since all types are callable in python.
    However this class overrides the error messages to make them more directed towards types
    """

    def __init__(self, _type, **kwargs):
        u"""
        :param _type: Any python type. Its constructor will be called with the value of the individual cell as its
            only argument. If it throws an exception, the value is considered to fail the validation, otherwise it has passed
        """
        if isinstance(_type, type):
            super(CanConvertValidation, self).__init__(_type, **kwargs)
        else:
            raise PanSchArgumentError(u'{} is not a valid type'.format(_type))

    @property
    def default_message(self):
        return u'cannot be converted to type {}'.format(self.callable)


class MatchesPatternValidation(SeriesValidation):
    u"""
    Validates that a string or regular expression can match somewhere in each element in this column
    """

    def __init__(self, pattern, options={}, **kwargs):
        u"""
        :param kwargs: Arguments to pass to Series.str.contains
            (http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.str.contains.html)
            pat is the only required argument
        """
        self.pattern = pattern
        self.options = options
        super(MatchesPatternValidation, self).__init__(**kwargs)

    @property
    def default_message(self):
        return u'does not match the pattern "{}"'.format(self.pattern)

    def validate(self, series):
        return series.astype('unicode').str.contains(self.pattern, **self.options)


class TrailingWhitespaceValidation(SeriesValidation):
    u"""
    Checks that there is no trailing whitespace in this column
    """

    def __init__(self, **kwargs):
        super(TrailingWhitespaceValidation, self).__init__(**kwargs)

    @property
    def default_message(self):
        return u'contains trailing whitespace'

    def validate(self, series):
        return ~series.astype(unicode).str.contains(u'\s+$')


class LeadingWhitespaceValidation(SeriesValidation):
    u"""
    Checks that there is no leading whitespace in this column
    """

    def __init__(self, **kwargs):
        super(LeadingWhitespaceValidation, self).__init__(**kwargs)

    @property
    def default_message(self):
        return u'contains leading whitespace'

    def validate(self, series):
        return ~series.astype(unicode).str.contains(u'^\s+')


class InListValidation(SeriesValidation):
    u"""
    Checks that each element in this column is contained within a list of possibilities
    """

    def __init__(self, options, case_sensitive = True, **kwargs):
        u"""
        :param options: A list of values to check. If the value of a cell is in this list, it is considered to pass the
            validation
        """
        self.case_sensitive = case_sensitive
        self.options = options
        super(InListValidation, self).__init__(**kwargs)

    @property
    def default_message(self):
        return u'is not in the list of legal options ({})'.format(u', '.join(self.options))

    def validate(self, series):
        if self.case_sensitive:
            return series.isin(self.options)
        else:
            return series.unicode.lower().isin([s.lower() for s in self.options])


class DateFormatValidation(SeriesValidation):
    u"""
    Checks that each element in this column is a valid date according to a provided format string
    """

    def __init__(self, date_format, **kwargs):
        u"""
        :param date_format: The date format string to validate the column against. Refer to the date format code
            documentation at https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior for a full
            list of format codes
        """
        self.date_format = date_format
        super(DateFormatValidation, self).__init__(**kwargs)

    @property
    def default_message(self):
        return u'does not match the date format string "{}"'.format(self.date_format)

    def valid_date(self, val):
        try:
            datetime.datetime.strptime(val, self.date_format)
            return True
        except:
            return False

    def validate(self, series):
        return series.astype(unicode).apply(self.valid_date)

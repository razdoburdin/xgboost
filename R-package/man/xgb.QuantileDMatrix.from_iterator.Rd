% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/xgb.DMatrix.R
\name{xgb.QuantileDMatrix.from_iterator}
\alias{xgb.QuantileDMatrix.from_iterator}
\title{QuantileDMatrix from External Data}
\usage{
xgb.QuantileDMatrix.from_iterator(
  data_iterator,
  missing = NA,
  nthread = NULL,
  ref = NULL,
  max_bin = NULL
)
}
\arguments{
\item{data_iterator}{A data iterator structure as returned by \code{\link[=xgb.DataIter]{xgb.DataIter()}},
which includes an environment shared between function calls, and functions to access
the data in batches on-demand.}

\item{missing}{A float value to represents missing values in data.

Note that, while functions like \code{\link[=xgb.DMatrix]{xgb.DMatrix()}} can take a generic \code{NA} and interpret it
correctly for different types like \code{numeric} and \code{integer}, if an \code{NA} value is passed here,
it will not be adapted for different input types.

For example, in R \code{integer} types, missing values are represented by integer number \code{-2147483648}
(since machine 'integer' types do not have an inherent 'NA' value) - hence, if one passes \code{NA},
which is interpreted as a floating-point NaN by \code{\link[=xgb.ExternalDMatrix]{xgb.ExternalDMatrix()}} and by
\code{\link[=xgb.QuantileDMatrix.from_iterator]{xgb.QuantileDMatrix.from_iterator()}}, these integer missing values will not be treated as missing.
This should not pose any problem for \code{numeric} types, since they do have an inheret NaN value.}

\item{nthread}{Number of threads used for creating DMatrix.}

\item{ref}{The training dataset that provides quantile information, needed when creating
validation/test dataset with \code{\link[=xgb.QuantileDMatrix]{xgb.QuantileDMatrix()}}. Supplying the training DMatrix
as a reference means that the same quantisation applied to the training data is
applied to the validation/test data}

\item{max_bin}{The number of histogram bin, should be consistent with the training parameter
\code{max_bin}.

This is only supported when constructing a QuantileDMatrix.}
}
\value{
An 'xgb.DMatrix' object, with subclass 'xgb.QuantileDMatrix'.
}
\description{
Create an \code{xgb.QuantileDMatrix} object (exact same class as would be returned by
calling function \code{\link[=xgb.QuantileDMatrix]{xgb.QuantileDMatrix()}}, with the same advantages and limitations) from
external data supplied by \code{\link[=xgb.DataIter]{xgb.DataIter()}}, potentially passed in batches from
a bigger set that might not fit entirely in memory, same way as \code{\link[=xgb.ExternalDMatrix]{xgb.ExternalDMatrix()}}.

Note that, while external data will only be loaded through the iterator (thus the full data
might not be held entirely in-memory), the quantized representation of the data will get
created in-memory, being concatenated from multiple calls to the data iterator. The quantized
version is typically lighter than the original data, so there might be cases in which this
representation could potentially fit in memory even if the full data does not.

For more information, see the guide 'Using XGBoost External Memory Version':
\url{https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html}
}
\seealso{
\code{\link[=xgb.DataIter]{xgb.DataIter()}}, \code{\link[=xgb.DataBatch]{xgb.DataBatch()}}, \code{\link[=xgb.ExternalDMatrix]{xgb.ExternalDMatrix()}},
\code{\link[=xgb.QuantileDMatrix]{xgb.QuantileDMatrix()}}
}

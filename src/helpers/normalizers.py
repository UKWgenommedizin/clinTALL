from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class LogCPMNormalizer(BaseEstimator, TransformerMixin):
    """
    Normalize RNA-seq count data using log2(CPM + pseudocount).
    """

    def __init__(
        self,
        pseudocount=1.0,
        center=False,
        scale=False,
        eps=1e-8
    ):
        """
        Parameters
        ----------
        pseudocount : float
            Value added before log2 transform.
        center : bool
            Whether to mean-center genes (using training data).
        scale : bool
            Whether to scale genes to unit variance (training data).
        eps : float
            Small value to avoid division by zero.
        """
        self.pseudocount = pseudocount
        self.center = center
        self.scale = scale
        self.eps = eps

        self.libsize_ = None
        self.gene_mean_ = None
        self.gene_std_ = None
        self.fitted_ = False

    def fit(self, X, y=None):
        """
        Fit the normalizer on training data.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Raw count matrix (samples x genes).
        """
        X = self._check_input(X)
        #print(X.head())
        #print(X.dtypes)

        libsize = X.sum(axis=1).values
        libsize[libsize == 0] = 1.0
        self.libsize_ = libsize
        
        cpm = (X.T / libsize).T * 1e6
        logcpm = np.log2(cpm + self.pseudocount)
        
        if self.center:
            self.gene_mean_ = logcpm.mean(axis=0).values  # Convert to numpy
        if self.scale:
            gene_std = logcpm.std(axis=0).values  # Convert to numpy
            gene_std[gene_std == 0] = self.eps  # Replace zeros
            self.gene_std_ = gene_std
        
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Transform data using training-derived parameters.
        """
        if not self.fitted_:
            raise RuntimeError("LogCPMNormalizer must be fitted before transform().")

        X = self._check_input(X)
        #print(X.head())
        #print(X.dtypes)
        
        # Library sizes (test samples)
        libsize = X.sum(axis=1).values
        libsize[libsize == 0] = 1.0

        # CPM
        cpm = (X.T / libsize).T * 1e6

        # log2(CPM + pseudocount)
        logcpm = np.log2(cpm + self.pseudocount)

        if self.center:
            logcpm = logcpm - self.gene_mean_

        if self.scale:
            logcpm = logcpm / self.gene_std_

        #print(logcpm.head())
        return logcpm.values

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    
    @staticmethod
    def _check_input(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        X = X.copy()
        
        # Check all columns are numeric
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            try:
                X[non_numeric_cols] = X[non_numeric_cols].apply(pd.to_numeric, errors='raise')
                print(f"Converted {len(non_numeric_cols)} columns to numeric types.")
            except Exception as e:
                raise ValueError(f"Input count data must be numeric. Failed on columns: {non_numeric_cols.tolist()}") from e
        
        # Check that all values are non-negative
        if (X < 0).any().any():
            n_neg = (X < 0).sum().sum()
            raise ValueError(f"Input count data must be non-negative. Found {n_neg} negative values.")
        
        # Check that all values are integers (allowing for float representation like 1.0, 2.0)
        if not np.allclose(X.values, np.round(X.values), rtol=0, atol=1e-10):
            non_int_mask = ~np.isclose(X.values, np.round(X.values), rtol=0, atol=1e-10)
            n_non_int = non_int_mask.sum()
            example_vals = X.values[non_int_mask].flatten()[:5]
            raise ValueError(f"Input count data must be integers. Found {n_non_int} non-integer values. "
                            f"Examples: {example_vals}")
        
        # Explicitly convert to integers
        X = X.round().astype(np.int64)
        
        return X
"""    @staticmethod
    def _check_input(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        # check all values are numeric
        if not np.issubdtype(X.dtypes.values.dtype, np.number):
            # try to convert 
            try:
                X = X.apply(pd.to_numeric, errors='raise')
                print("Converted input to numeric types. May cause data loss if non-numeric values were present.")
            except Exception as e:
                raise ValueError("Input count data must be numeric") from e
        # check that all values are non-negative
        if (X < 0).any().any():
            raise ValueError("Input count data must be non-negative")
        #check that all values are integers
        if not np.all(np.equal(np.mod(X.values, 1), 0)):
            raise ValueError("Input count data must be integers")
        
        return X.copy()"""
    
class DESeqNormalizer(BaseEstimator, TransformerMixin):
    """
    Normalize count data using DESeq's variance stabilizing transformation (VST).
    Implements scikit-learn's fit/transform interface.
    
    This is a wrapper for DESeq's normalization method that follows
    scikit-learn's transformer interface.
    """
    
    def __init__(self, design_column='Subgroup_Name', use_dummy=False):
        """
        Initialize the DESeqNormalizer.
        
        Parameters
        ----------
        design_column : str, default='Subgroup Name'
            The column name in metadata that contains the design information.
        """
        self.design_column = design_column
        self.fitted = None
        self.vst_counts_ = None
        self.dds_ = None
        self.use_dummy = use_dummy
    
    def fit(self, X, y=None):
        """
        Fit the DESeq normalizer.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The count data to be normalized.
        y : pandas.DataFrame or pandas.Series
            The metadata containing the design column.
            
        Returns
        -------
        self : object
            Returns self.
        """
        if y is None and not self.use_dummy:
            raise ValueError("Metadata (y) is required for DESeq normalization")
        elif y is None and self.use_dummy:
            # Create dummy metadata
            y = pd.DataFrame({self.design_column: ['A'] * X.shape[0]}, index=X.index)
        
        
        # Create a copy to avoid modifying the original
        counts = X.copy()
        
        if self.design_column in counts.columns:
            counts = counts.drop(columns=[self.design_column])
            
        # Import DeseqDataSet here to make the dependency optional
        try:
            from pydeseq2.dds import DeseqDataSet
        except ImportError:
            raise ImportError("DeseqDataSet is required for DESeq normalization")
        
        # Create DeseqDataSet and apply VST
        
        self.dds_ = DeseqDataSet(counts=counts, metadata=y, design=self.design_column)
        self.dds_.vst_fit()
        self.fitted = True
        
    
    def transform(self, X, y=None):
        """
        Transform the data using the fitted DESeq normalizer.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The count data to be normalized.
        y : pandas.DataFrame or pandas.Series, optional
            Not used, present for API consistency.
            
        Returns
        -------
        pandas.DataFrame
            The normalized data.
        """
        # Check if fit has been called
        if self.fitted is None:
            raise ValueError("This DESeqNormalizer instance is not fitted yet. "
                             "Call 'fit' before using this estimator.")
        
     
        
        # Use the stored vst_counts_ for transformation
        return self.dds_.vst_transform(counts=X)
    
    def fit_transform(self, X, y, fit_type='parametric'):
        """
        Fit the normalizer and return the transformed data.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The count data to be normalized.
        y : pandas.DataFrame or pandas.Series
            The metadata containing the design column.
            
        Returns
        -------
        pandas.DataFrame
            The normalized data.
        """
        try:
            from pydeseq2.dds import DeseqDataSet
        except ImportError:
            raise ImportError("DeseqDataSet is required for DESeq normalization")
        
        if y is None:
            if self.design_column in X.columns:
                print('extracting metadata from X for DESeq normalization')
                y=X[self.design_column]
                X = X.drop(columns=[self.design_column])
            elif self.use_dummy:
                print('using dummy metadata for DESeq normalization')
                # Create dummy metadata
                y = pd.DataFrame({self.design_column: ['A'] * X.shape[0]}, index=X.index)
            else:
                raise ValueError("Metadata (y) is required for DESeq normalization")
        
        # Create a copy to avoid modifying the original
        counts = X.copy()
        
        if self.design_column in counts.columns:
            counts = counts.drop(columns=[self.design_column])

        #dds = DeseqDataSet(counts=counts, metadata=y, design=self.design_column)
        self.dds_ = DeseqDataSet(counts=counts, metadata=y, design=self.design_column)
        self.dds_.vst_fit()
        self.fitted = True
        return self.dds_.vst_transform(counts = counts)


class RankNormalizer(BaseEstimator, TransformerMixin):
    """
    Normalize data by ranking values and dividing by mean rank.
    Implements scikit-learn's fit/transform interface.
    """
    
    def __init__(self, axis=1, ascending=False, na_option='keep'):
        """
        Initialize the RankNormalizer.
        
        Parameters
        ----------
        axis : int, default=1
            The axis along which to rank.
            0 for ranking along columns, 1 for ranking along rows.
        ascending : bool, default=False
            Whether to rank in ascending order.
        na_option : str, default='keep'
            How to handle NaN values.
        """
        self.axis = axis
        self.ascending = ascending
        self.na_option = na_option
        self.mean_rank_ = None
    
    def fit(self, X, y=None):
        """
        Fit the rank normalizer by computing the mean rank.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The data to be normalized.
        y : None
            Ignored, present for API consistency.
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Convert to DataFrame if not already
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # Compute ranks
        ranked = X_df.rank(axis=self.axis, na_option=self.na_option, 
                           ascending=self.ascending)
        
        # Store mean rank
        self.mean_rank_ = ranked.mean()
        
        return self
    
    def transform(self, X, y=None):
        """
        Transform the data using the fitted rank normalizer.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The data to be normalized.
        y : None
            Ignored, present for API consistency.
            
        Returns
        -------
        pandas.DataFrame
            The normalized data.
        """
        # Check if fit has been called
        if self.mean_rank_ is None:
            raise ValueError("This RankNormalizer instance is not fitted yet. "
                             "Call 'fit' before using this estimator.")
        
        # Convert to DataFrame if not already
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # Compute ranks
        ranked = X_df.rank(axis=self.axis, na_option=self.na_option, 
                          ascending=self.ascending)
        
        # Normalize by mean rank
        return ranked / self.mean_rank_
    
    def fit_transform(self, X, y):
        """
        Fit the normalizer and return the transformed data.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The data to be normalized.
        y : None
            Ignored, present for API consistency.
            
        Returns
        -------
        pandas.DataFrame
            The normalized data.
        """
        return self.fit(X).transform(X)

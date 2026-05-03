"""
Exception classes for PyGAD.
"""


class GeneConstraintError(Exception):
    """
    Raised when no valid gene value can be found that satisfies the gene constraint.
    
    Attributes:
        gene_index (int): The index of the gene that failed the constraint.
        gene_value (Any): The current value of the gene.
        solution (list): The solution/chromosome containing the gene.
        sample_size_used (int): The sample size used in the attempt.
        generations_completed (int): Number of generations completed when the error occurred.
        details (str): Additional diagnostic information.
    """
    
    def __init__(self, gene_index, gene_value, solution=None, sample_size_used=None, 
                 generations_completed=None, details=""):
        self.gene_index = gene_index
        self.gene_value = gene_value
        self.solution = solution
        self.sample_size_used = sample_size_used
        self.generations_completed = generations_completed
        self.details = details
        
        message = (f"Failed to find a valid gene value at index {gene_index} "
                  f"(current value: {gene_value}). "
                  f"Sample size used: {sample_size_used}. "
                  f"Generations completed: {generations_completed}. "
                  f"Details: {details}")
        super().__init__(message)


class DuplicateGeneError(Exception):
    """
    Raised when no unique gene value can be found (allow_duplicate_genes=False).
    
    Attributes:
        gene_index (int): The index of the gene that has duplicate values.
        gene_value (Any): The current value of the gene.
        solution (list): The solution/chromosome containing the gene.
        sample_size_used (int): The sample size used in the attempt.
        generations_completed (int): Number of generations completed when the error occurred.
        details (str): Additional diagnostic information.
    """
    
    def __init__(self, gene_index, gene_value, solution=None, sample_size_used=None,
                 generations_completed=None, details=""):
        self.gene_index = gene_index
        self.gene_value = gene_value
        self.solution = solution
        self.sample_size_used = sample_size_used
        self.generations_completed = generations_completed
        self.details = details
        
        message = (f"Failed to find a unique gene value at index {gene_index} "
                  f"(current value: {gene_value}). "
                  f"Sample size used: {sample_size_used}. "
                  f"Generations completed: {generations_completed}. "
                  f"Details: {details}")
        super().__init__(message)

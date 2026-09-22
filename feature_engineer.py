class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()

    def add_total_pages(self):
        self.df['TotalPages'] = (
            self.df['Administrative'] +
            self.df['Informational'] +
            self.df['ProductRelated']
        )
        return self

    def add_total_duration(self):
        self.df['TotalDuration'] = (
            self.df['Administrative_Duration'] +
            self.df['Informational_Duration'] +
            self.df['ProductRelated_Duration']
        )
        return self

    def add_product_focus_ratio(self):
        self.df['ProductFocusRatio'] = (
            self.df['ProductRelated'] / self.df['TotalPages'].replace(0, 1)
        )
        return self

    def get_dataframe(self):
        return self.df
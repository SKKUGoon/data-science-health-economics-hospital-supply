* For each hospitals
* Create data cleaning 
  * Strip str columns
* Feature engineering to create these columns
  * id: str
  * date: datetime
  * sex: Literal['male', 'female', 'NA']
  * age: int
  * department: str
  * primary_diagnosis: str
  * secondary_diagnosis: str
  * prescription: str

* 'load' function should take `root_dir: Path` and **kwargs as an input, and `pd.DataFrame` as an output.
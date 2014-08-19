import gzip

class MalletAssignment:

  def __init__(self, line, debug=False):
    if debug:
      for ii in xrange(len(line.split())):
        print ii, line.split()[ii]
        
    self.doc, _, self.index, self.term_id, self.term, self.assignment = line.split()
    self.doc = int(self.doc)
    self.index = int(self.index)
    self.term_id = int(self.term_id)
    self.assignment = int(self.assignment)


class MalletStateBuffer:

  def __init__(self, filename):
    self.infile = gzip.GzipFile(filename)
    self.current_doc = -1
    self.buffer = []

  def __iter__(self):
    for ii in self.infile:
      if ii.startswith("#"):
        continue
      line = MalletAssignment(ii)

      if line.doc != self.current_doc and self.buffer:
        yield self.buffer
        self.buffer = []
      self.current_doc = line.doc

      self.buffer.append(line)
    yield self.buffer


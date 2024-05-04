#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <stack>
#include <memory.h>
#include <string>
#include <sstream>
#include <exception>
#include <stdexcept>
#include <algorithm>

std::mt19937 random;

struct Node {
  int val, h = 1;
  Node *r_child = nullptr, *l_child = nullptr, *p = nullptr;
  Node(int value) : val(value) {
  }
};

class AVLTree {
 public:
  AVLTree() {
    tnull = new Node(0);
    root = tnull;
    tnull->h = 0;
  }

  void Insert(int val, Node* cur) {
    if (root == tnull) {
      root = new Node(val);
      root->l_child = root->p = root->r_child = tnull;
      return;
    }

    if (val < cur->val) {
      if (cur->l_child == tnull) {
        auto node = cur->l_child = new Node(val);
        node->l_child = node->r_child = tnull;
        node->p = cur;
      } else {
        Insert(val, cur->l_child);
      }
    } else {
      if (cur->r_child == tnull) {
        auto node = cur->r_child = new Node(val);
        node->l_child = node->r_child = tnull;
        node->p = cur;
      } else {
        Insert(val, cur->r_child);
      }
    }

    RecountHieght(cur);
    if (std::abs(Delta(cur)) == 2) {
      Rebalance(cur);
    }
  }

  void Print(Node* cur) {
    if (cur == tnull) {
      return;
    }

    Print(cur->l_child);
    std::cout << cur->val << ' ';
    Print(cur->r_child);
  }

  Node* Root() {
    return root;
  }

 private:
  Node* tnull;
  Node* root;

  void Rebalance(Node* node) {
    if (node == tnull)
      return;

    if (Delta(node) == -2) {
      int d = Delta(node->r_child);
      if (d == 0 || d == -1) {
        LeftSmallRot(node);
      } else if (d == 1) {
        LeftBigRot(node);
      }
    } else if (Delta(node) == 2) {
      int d = Delta(node->l_child);
      if (d == 1 || d == 0) {
        RightSmallRot(node);
      } else if (d == -1) {
        RightBigRot(node);
      }
    }
  }

  int Delta(Node* node) {
    return node->l_child->h - node->r_child->h;
  }

  void LeftSmallRot(Node* node) {
    auto b = node->r_child;
    node->r_child = b->l_child;
    b->l_child->p = node;
    b->l_child = node;
    GetReferenceFromParent(node) = b;
    b->p = node->p;
    node->p = b;
    RecountHieght(node);
    RecountHieght(b);
    if (b->p == tnull) {
      root = b;
    }
  }

  Node*& GetReferenceFromParent(Node* node) {

    if (node->p->l_child == node) {
      return node->p->l_child;
    } else {
      return node->p->r_child;
    }
  }

  void LeftBigRot(Node* node) {
    RightSmallRot(node->r_child);
    LeftSmallRot(node);
  }

  void RightSmallRot(Node* node) {
    auto b = node->l_child;
    node->l_child = b->r_child;
    b->r_child->p = node;
    b->r_child = node;
    GetReferenceFromParent(node) = b;
    b->p = node->p;
    node->p = b;
    RecountHieght(node);
    RecountHieght(b);
    if (b->p == tnull) {
      root = b;
    }
  }

  void RightBigRot(Node* node) {
    LeftSmallRot(node->l_child);
    RightSmallRot(node);
  }

  void RecountHieght(Node* node) {
    if (node == tnull)
      return;
    node->h = std::max(node->l_child->h, node->r_child->h) + 1;
  }
};

struct CTNode {
  int k, pr;
  CTNode *par = nullptr, *left = nullptr, *right = nullptr;
  CTNode(int key, int priority) : k(key), pr(priority) {
  }
};

void CTPrint(CTNode* node) {
  if (node == nullptr)
    return;
  CTPrint(node->left);
  std::cout << node->k << ' ';
  CTPrint(node->right);
}

CTNode* BuildCartesianTree(std::vector<int> arr) {
  std::stack<CTNode*> stack;

  for (int i = 0; i < arr.size(); ++i) {
    CTNode* node = new CTNode(arr[i], random());

    CTNode* last = nullptr;
    while (!stack.empty() && stack.top()->pr > node->pr) {
      last = stack.top();
      stack.pop();
    }

    if (!stack.empty()) {
      stack.top()->right = node;
    }

    stack.push(node);
    node->left = last;
  }

  while (stack.size() > 1) {
    stack.pop();
  }

  return stack.top();
}

CTNode* Merge(CTNode* left, CTNode* right) {
  if (left == nullptr)
    return right;
  if (right == nullptr)
    return left;

  if (left->pr < right->pr) {
    left->right = Merge(left->right, right);
    return left;
  } else {
    right->left = Merge(left, right->left);
    return right;
  }
}

std::pair<CTNode*, CTNode*> Split(CTNode* node, int key) {
  if (node == nullptr)
    return {nullptr, nullptr};

  if (node->k > key) {
    auto tmp = Split(node->left, key);
    node->left = tmp.second;
    return {tmp.first, node};
  } else {
    auto tmp = Split(node->right, key);
    node->right = tmp.first;
    return {node, tmp.second};
  }
}

CTNode* CTInsert(CTNode* tree, int key) {
  auto tmp = Split(tree, key);

  // CTPrint(tmp.first);
  // std::cout << std::endl;
  // CTPrint(tmp.second);
  // std::cout << std::endl;

  auto node = new CTNode(key, random());
  return Merge(Merge(tmp.first, node), tmp.second);
}

CTNode* CTErase(CTNode* node, int key) {
  auto splittedKey = Split(node, key);
  auto left = Split(splittedKey.first, key - 1);
  return Merge(left.first, splittedKey.second);
}

// По неявному ключу
struct UCTNode {
  int size = 1;
  int val, pr, sum;
  UCTNode *par = nullptr, *left = nullptr, *right = nullptr;
  UCTNode(int value, int priority) : val(value), pr(priority), sum(val) {
  }

  void Recount() {
    size = 1;
    sum = val;
    if (left) {
      size += left->size;
      sum += left->sum;
    }
    if (right) {
      size += right->size;
      sum += right->sum;
    }
  }

  int Ind() {
    int ind = 0;
    if (left)
      ind += left->size;
    return ind;
  }
};

void UCTPrint(UCTNode* node) {
  if (node == nullptr)
    return;
  UCTPrint(node->left);
  std::cout << node->val << ' ';
  UCTPrint(node->right);
}

UCTNode* BuildUnCartesianTree(std::vector<int> arr) {
  std::stack<UCTNode*> stack;

  for (int i = 0; i < arr.size(); ++i) {
    UCTNode* node = new UCTNode(arr[i], random());

    UCTNode* last = nullptr;
    while (!stack.empty() && stack.top()->pr > node->pr) {
      last = stack.top();
      stack.pop();
    }

    if (!stack.empty()) {
      stack.top()->right = node;
      stack.top()->Recount();
    }

    stack.push(node);
    node->left = last;
    node->Recount();
  }

  while (stack.size() > 1) {
    stack.pop();
  }

  return stack.top();
}

UCTNode* Merge(UCTNode* left, UCTNode* right) {
  if (left == nullptr)
    return right;
  if (right == nullptr)
    return left;

  if (left->pr < right->pr) {
    left->right = Merge(left->right, right);
    left->Recount();
    return left;
  } else {
    right->left = Merge(left, right->left);
    right->Recount();
    return right;
  }
}

std::pair<UCTNode*, UCTNode*> Split(UCTNode* node, int pos) {
  if (node == nullptr)
    return {nullptr, nullptr};

  if (node->Ind() + 1 > pos) {
    auto tmp = Split(node->left, pos);
    node->left = tmp.second;
    node->Recount();
    return {tmp.first, node};
  } else {
    auto tmp = Split(node->right, pos - node->Ind() - 1);
    node->right = tmp.first;
    node->Recount();
    return {node, tmp.second};
  }
}

UCTNode* UCTInsert(UCTNode* tree, int val, int pos) {
  auto tmp = Split(tree, pos + 1);

  // CTPrint(tmp.first);
  // std::cout << std::endl;
  // CTPrint(tmp.second);
  // std::cout << std::endl;

  auto node = new UCTNode(val, random());
  return Merge(Merge(tmp.first, node), tmp.second);
}

int UCTSum(UCTNode* tree, int l, int r) {
  auto splitted_l = Split(tree, l);
  auto splitted_r = Split(splitted_l.second, r - l);
  int res = splitted_r.first->sum;
  Merge(splitted_l.first, Merge(splitted_r.first, splitted_r.second));
  return res;
}

// Long arithmethics.
class LongInteger;

bool operator<(const LongInteger& lhs, const LongInteger& rhs);

bool operator>(const LongInteger& lhs, const LongInteger& rhs);
bool operator>=(const LongInteger& lhs, const LongInteger& rhs);

bool operator<=(const LongInteger& lhs, const LongInteger& rhs);

LongInteger operator+(const LongInteger& lhs, const LongInteger& rhs);

LongInteger operator-(const LongInteger& lhs, const LongInteger& rhs);

LongInteger operator*(const LongInteger& lhs, const LongInteger& rhs);

class LongInteger {
 public:
  std::string ToString() const {
    std::string res;

    if (sign_ == -1)
      res += '-';

    res += std::to_string((digits_.size() > 0 ? digits_.back() : 0));

    for (int i = (int)digits_.size() - 2; i >= 0; --i) {
      std::string str = std::to_string(digits_[i]);
      res += std::string(9 - str.size(), '0') + str;
    }

    return res;
  }

  LongInteger(std::string str) {
    sign_ = str[0] == '-' ? -1 : 1;
    int add = sign_ == -1;

    for (int i = str.size(); i > add; i -= 9) {
      if (i <= 9 + add) {
        digits_.push_back(std::stoi(str.substr(add, i - add)));
      } else {
        digits_.push_back(std::stoi(str.substr(i - 9, 9)));
      }
    }

    while (digits_.size() > 0 && digits_.back() == 0)
      digits_.pop_back();
  }

  LongInteger(long long num) {
    digits_.push_back(num % base_);
    if (num >= base_)
      digits_.push_back(num / base_);
  }
  LongInteger() = default;

  LongInteger& operator+=(const LongInteger& rhs) {
    if (rhs.sign_ == -1 && sign_ == 1) {
      return *this > -rhs ? *this -= -rhs : *this = -(-rhs - *this);
    } else if (sign_ == -1 && rhs.sign_ == 1) {
      return -*this > rhs ? *this = -*this - rhs : *this = rhs - (-*this);
    }

    int carry = 0;
    for (size_t i = 0; i < std::max(digits_.size(), rhs.digits_.size()) || carry; ++i) {
      if (i == digits_.size()) {
        digits_.push_back(0);
      }

      int add = (i < rhs.digits_.size() ? rhs.digits_[i] : 0) + carry;
      digits_[i] += (i < rhs.digits_.size() ? rhs.digits_[i] : 0) + carry;

      carry = digits_[i] >= base_;
      if (carry) {
        digits_[i] -= base_;
      }
    }

    return *this;
  }

  LongInteger& operator-=(const LongInteger& rhs) {
    if (rhs.sign_ == -1 && sign_ == -1) {
      return *this += -rhs;
    } else if (rhs.sign_ == -1 && sign_ == 1) {
      return *this += -rhs;
    } else if (rhs.sign_ == 1 && sign_ == -1) {
      return *this = rhs + -*this;
    }

    int carry = 0;
    for (size_t i = 0; i < rhs.digits_.size() || carry; ++i) {
      digits_[i] -= (i < rhs.digits_.size() ? rhs.digits_[i] : 0) + carry;

      carry = digits_[i] < 0;
      if (carry) {
        digits_[i] += base_;
      }
    }

    while (digits_.size() > 0 && digits_.back() == 0)
      digits_.pop_back();

    return *this;
  }

  LongInteger& operator*=(const LongInteger& rhs) {
    std::vector<int> res(rhs.digits_.size() + digits_.size() + 1);
    for (int i = 0; i < rhs.digits_.size(); ++i) {
      for (int j = 0, carry = 0; j < digits_.size() || carry; ++j) {
        long long cur = res[i + j] + rhs.digits_[i] * 1ll * (j < digits_.size() ? digits_[j] : 0) + carry;
        res[i + j] = cur % base_;
        carry = cur / base_;
      }
    }

    sign_ *= rhs.sign_;

    while (res.size() > 0 && res.back() == 0)
      res.pop_back();

    digits_ = res;

    return *this;
  }

  LongInteger operator-() const {
    LongInteger res(*this);
    res.sign_ *= -1;
    return res;
  }

  friend bool operator<(const LongInteger&, const LongInteger&);

 private:
  std::vector<int> digits_;
  int base_ = 1000'000'000, sign_ = 1;
};

std::ostream& operator<<(std::ostream& out, const LongInteger& num) {
  return out << num.ToString();
}

bool operator<(const LongInteger& lhs, const LongInteger& rhs) {
  if (lhs.sign_ == -1 && rhs.sign_ == 1) {
    return true;
  } else if (lhs.sign_ == 1 && rhs.sign_ == -1) {
    return false;
  }

  for (int i = std::max(lhs.digits_.size(), rhs.digits_.size()); i >= 0; --i) {
    int l = i < lhs.digits_.size() ? lhs.digits_[i] : 0;
    int r = i < rhs.digits_.size() ? rhs.digits_[i] : 0;

    if (l < r)
      return lhs.sign_ == 1;
    else if (l > r)
      return lhs.sign_ != 1;
  }
}

bool operator>(const LongInteger& lhs, const LongInteger& rhs) {
  return rhs < lhs;
}

bool operator>=(const LongInteger& lhs, const LongInteger& rhs) {
  return !(rhs < lhs);
}

bool operator<=(const LongInteger& lhs, const LongInteger& rhs) {
  return !(rhs > lhs);
}

LongInteger operator+(const LongInteger& lhs, const LongInteger& rhs) {
  LongInteger res = lhs;
  res += rhs;
  return res;
}

LongInteger operator-(const LongInteger& lhs, const LongInteger& rhs) {
  LongInteger res = lhs;
  res -= rhs;
  return res;
}

LongInteger operator*(const LongInteger& lhs, const LongInteger& rhs) {
  LongInteger res = lhs;
  res *= rhs;
  return res;
}

LongInteger operator*(const LongInteger& lhs, int rhs) {
  LongInteger res = lhs;
  res *= LongInteger(rhs);
  return res;
}

std::istream& operator>>(std::istream& in, LongInteger& num) {
  std::string tmp;
  in >> tmp;
  num = LongInteger(tmp);
  return in;
}

// My vector
class Vector {
 public:
  Vector(int count) : arr_(new int[count]), size_(count), capacity_(count) {
    memset(arr_, 0, size_ * sizeof(int));
  }

  Vector(const Vector& other) {
    size_ = other.size_;
    capacity_ = other.capacity_;
    memcpy(arr_, other.arr_, sizeof arr_);
  }

  Vector& operator=(Vector other) {
    Swap(other);
    return *this;
  }

  int& operator[](int index) {
    if (index >= size_ || index < 0) {
      throw std::out_of_range("Index out of vector range");
    }

    return arr_[index];
  }

  void PushBack(int val) {
    if (size_ == capacity_) {
      DoubleSize();
    }

    arr_[size_++] = val;
  }

  void PopBack() {
    if (size_ == 0) {
      throw std::out_of_range("PopBack() called on empty vector");
    }

    if (--size_ <= capacity_ / 4) {
      DecreaseSize();
    }
  }

  ~Vector() {
    delete[] arr_;
  }

 private:
  size_t size_, capacity_;
  int* arr_;

  void Swap(Vector& vector) {
    std::swap(size_, vector.size_);
    std::swap(capacity_, vector.capacity_);
    std::swap(arr_, vector.arr_);
  }

  void DoubleSize() {
    int* ptr = new int[capacity_ *= 2];
    memmove(ptr, arr_, sizeof(arr_));
    delete[] arr_;
    arr_ = ptr;
  }

  void DecreaseSize() {
    int* ptr = new int[capacity_ /= 2];
    memmove(ptr, arr_, size_ * sizeof(int));
    delete[] arr_;
    arr_ = ptr;
  }
};

template <int N, int M>
struct GCD {
  static constexpr int gcd = GCD<M, N % M>::gcd;
};

template <int N>
struct GCD<N, 0> {
  static constexpr int gcd = N;
};

template <int Num>
struct IsPrime {
 private:
  template <int N, int M = N - 1>
  struct HasNoDivisions {
    static constexpr bool has_no_divisions = N % M != 0 && HasNoDivisions<N, M - 1>::has_no_divisions;
  };

  template <int N>
  struct HasNoDivisions<N, 1> {
    static constexpr bool has_no_divisions = true;
  };

 public:
  static constexpr bool prime = HasNoDivisions<Num, Num - 1>::has_no_divisions;
};

// -------------------------------SECOND SEMESTER------------------------------- //

// Handy stuff
std::ostream& operator<<(std::ostream& stream, const std::vector<int>& a) {
  for (int val : a) {
    stream << val << ' ';
  }

  return stream;
}
using IntPair = std::pair<int, int>;

// --------SEGMENT TREE--------
struct STNode {
  int max = 0;
  int to_push = 0;
  int max_ind = 0;
};

class SegmentTree {
 public:
  SegmentTree(std::vector<int>& arr) : nodes(4 * arr.size()), n(arr.size()) {
    BuildTree(arr, 0, n, 0);
  }

  void Update(int x, int ind, int tl, int tr, int v) {
    if (tr - tl == 1) {
      nodes[v].max += x;
      return;
    }

    int tm = (tr + tl) / 2;
    if (ind >= tm) {
      Push(v, tl, tr);
      Update(x, ind, tm, tr, v * 2 + 2);
    } else {
      Push(v, tl, tr);
      Update(x, ind, tl, tm, v * 2 + 1);
    }

    Recount(v);
  }

  IntPair GetMax(int ql, int qr, int tl, int tr, int v) {
    if (ql > qr || tl > tr) {
      return {0, 0};
    }

    if (tl == ql && tr - 1 == qr) {
      return {nodes[v].max, nodes[v].max_ind};
    }

    int tm = (tr + tl) / 2;

    Push(v, tl, tr);
    auto l_res = GetMax(ql, std::min(tm - 1, qr), tl, tm, v * 2 + 1);
    auto r_res = GetMax(std::max(tm, ql), qr, tm, tr, v * 2 + 2);
    return l_res.first > r_res.first ? l_res : r_res;
  }

  void SegmentAddition(int addition, int ql, int qr, int tl, int tr, int v) {
    if (ql > qr || tl > tr)
      return;
    if (tl == ql && tr - 1 == qr) {
      nodes[v].max += addition;
      nodes[v].to_push += addition;
      return;
    }

    int tm = (tr + tl) / 2;

    Push(v, tl, tr);
    SegmentAddition(addition, ql, std::min(tm - 1, qr), tl, tm, v * 2 + 1);
    SegmentAddition(addition, std::max(tm, ql), qr, tm, tr, v * 2 + 2);
  }

 private:
  std::vector<STNode> nodes;
  int n;

  void BuildTree(std::vector<int>& arr, int tl, int tr, int v) {
    if (tr - tl == 1) {
      nodes[v].max = arr[tl];
      nodes[v].max_ind = tl;
      return;
    }

    int tm = (tr + tl) / 2;

    BuildTree(arr, tl, tm, v * 2 + 1);
    BuildTree(arr, tm, tr, v * 2 + 2);
    Recount(v);
  }

  void Recount(int v) {
    nodes[v].max = std::max(nodes[v * 2 + 1].max, nodes[v * 2 + 2].max);
    if (nodes[v].max == nodes[v * 2 + 1].max) {
      nodes[v].max_ind = nodes[v * 2 + 1].max_ind;
    } else {
      nodes[v].max_ind = nodes[v * 2 + 2].max_ind;
    }
  }

  void Push(int v, int tl, int tr) {
    if (nodes[v].to_push == 0)
      return;

    if (tr - tl == 1) {
      nodes[v].to_push = 0;
      return;
    }

    nodes[v * 2 + 1].to_push += nodes[v].to_push;
    nodes[v * 2 + 2].to_push += nodes[v].to_push;

    nodes[v * 2 + 1].max += nodes[v].to_push;
    nodes[v * 2 + 2].max += nodes[v].to_push;

    nodes[v].to_push = 0;
  }
};

// Biggest common sequence
std::vector<int> BiggestCommonSubsequence(std::vector<int> a, std::vector<int> b) {
  std::vector<std::vector<int>> dp(a.size() + 1, std::vector<int>(b.size() + 1, 0));
  auto path = dp;
  std::vector<int> answer;

  for (int i = 1; i <= a.size(); ++i) {
    for (int j = 1; j <= b.size(); ++j) {
      if (a[i - 1] == b[j - 1]) {
        dp[i][j] = 1 + dp[i - 1][j - 1];
        path[i][j] = 1;
      } else {
        if (dp[i - 1][j] >= dp[i][j - 1]) {
          path[i][j] = 2;
        } else {
          path[i][j] = 3;
        }
        dp[i][j] = std::max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
  }

  int i = a.size(), j = b.size();
  while (dp[i][j] > 0) {
    if (path[i][j] == 1) {
      answer.push_back(a[i - 1]);
      --i, --j;
    } else if (path[i][j] == 2) {
      i--;
    } else if (path[i][j] == 3) {
      j--;
    }
  }

  std::reverse(answer.begin(), answer.end());
  return answer;
}

// Biggest increasing subsequence
int BiggestIncreasingSequenceLength(std::vector<int> a) {
  std::vector<int> dp(a.size());
  auto path = dp;

  dp[0] = a[0];
  for (int i = 1; i < a.size(); ++i) {
    dp[i] = INT32_MAX;
  }

  for (int i = 1; i < a.size(); ++i) {
    auto it = std::upper_bound(dp.begin(), dp.end(), a[i]);
    *(++it) = a[i];
  }

  int max_length = 0;
  for (int i = 0; i < a.size(); ++i) {
    if (dp[i] != INT32_MAX) {
      max_length = i + 1;
    }
  }

  return max_length;
}

std::vector<int> BiggestIncreasingSequence(std::vector<int> a) {
  std::vector<int> dp(a.size(), 0);
  auto path = dp;

  std::vector<IntPair> sorted(a.size());
  for (int i = 0; i < a.size(); ++i) {
    sorted[i] = {a[i], i};
  }

  std::sort(sorted.begin(), sorted.end(),
            [](IntPair l, IntPair r) { return l.first == r.first ? l.second < r.second : l.first < r.first; });

  SegmentTree st(dp);
  for (int i = 0; i < sorted.size(); ++i) {
    int sorted_ind = sorted[i].second;
    auto max = st.GetMax(0, sorted_ind, 0, dp.size(), 0);
    st.Update(1 + max.first, sorted_ind, 0, dp.size(), 0);
    dp[sorted_ind] = 1 + max.first;
    path[sorted_ind] = max.first == 0 ? -1 : max.second;
  }

  std::vector<int> answer;
  IntPair max_length = st.GetMax(0, dp.size() - 1, 0, dp.size(), 0);
  int cur_ind = max_length.second;
  while (cur_ind != -1) {
    answer.push_back(a[cur_ind]);
    cur_ind = path[cur_ind];
  }

  std::reverse(answer.begin(), answer.end());
  return answer;
}

int main() {
  int n;
  std::cin >> n;

  std::vector<int> arr(n);
  for (int& val : arr) {
    std::cin >> val;
  }

  std::reverse(arr.begin(), arr.end());
  auto res = BiggestIncreasingSequence(arr);
  std::reverse(res.begin(), res.end());

  std::cout << res;
}

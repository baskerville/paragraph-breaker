const FITNESS_DEMERITS: u32  =  3_000;
const FLAGGED_DEMERITS: u32  =    100;
const LINE_DEMERITS: u32     =     10;

pub const INFINITE_PENALTY: i32  = 10_000;

const NULL: usize = ::std::usize::MAX;

#[derive(Debug, Copy, Clone)]
pub enum Item<T> {
    Box {
        width: i32,
        data: T,
    },
    Glue {
        width: i32,
        stretch: i32,
        shrink: i32,
    },
    Penalty {
        width: i32,
        penalty: i32,
        flagged: bool,
    },
}

impl<T> Item<T> {
    #[inline]
    pub fn is_box(&self) -> bool {
        match *self {
            Item::Box { .. } => true,
            _ => false
        }
    }

    #[inline]
    pub fn penalty(&self) -> i32 {
        match *self {
            Item::Penalty { penalty, .. } => penalty,
            _ => 0,
        }
    }

    #[inline]
    pub fn flagged(&self) -> bool {
        match *self {
            Item::Penalty { flagged, .. } => flagged,
            _ => false,
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct Node {
    position: usize,
    line: usize,
    sums: Sums,
    ratio: f32,
    width: i32,
    demerits: u32,
    fitness_class: usize,
    best_from: usize,
    next: usize,
}

impl Default for Node {
    fn default() -> Self {
        Node {
            position: 0,
            line: 0,
            sums: Sums::default(),
            ratio: 0.0,
            width: 0,
            demerits: 0,
            fitness_class: 1,
            best_from: NULL,
            next: NULL,
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct Sums {
    width: i32,
    stretch: i32,
    shrink: i32,
}

impl Default for Sums {
    fn default() -> Self {
        Sums {
            width: 0,
            stretch: 0,
            shrink: 0,
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct Candidate {
    demerits: u32,
    address: usize,
    ratio: f32,
    width: i32,
}

#[derive(Debug, Copy, Clone)]
pub struct Breakpoint {
    pub position: usize,
    pub ratio: f32,
    pub width: i32,
}

#[inline]
fn ratio<T>(ideal_len: i32, sums: &Sums, previous_sums: &Sums, item: &Item<T>) -> (f32, i32) {
    let mut actual_len = sums.width - previous_sums.width;

    if let Item::Penalty { width, .. } = *item {
        actual_len += width;
    }

    if actual_len < ideal_len {
        let stretch = sums.stretch - previous_sums.stretch;
        if stretch > 0 {
            ((ideal_len - actual_len) as f32 / stretch as f32, actual_len)
        } else {
            (::std::f32::INFINITY, actual_len)
        }
    } else if actual_len > ideal_len {
        let shrink = sums.shrink - previous_sums.shrink;
        if shrink > 0 {
            ((ideal_len - actual_len) as f32 / shrink as f32, actual_len)
        } else {
            (::std::f32::NEG_INFINITY, actual_len)
        }
    } else {
        (0.0, actual_len)
    }
}

#[inline]
fn fitness_class(ratio: f32) -> usize {
    if ratio < -0.5 {
        0
    } else if ratio <= 0.5 {
        1
    } else if ratio <= 1.0 {
        2
    } else {
        3
    }
}

#[inline]
fn badness(ratio: f32) -> u32 {
    (100.0 * ratio.abs().powi(3)) as u32
}

#[inline]
fn demerits<T>(ratio: f32, class: usize, active: &Node, item: &Item<T>, from_item: &Item<T>) -> u32 {
    let mut d = (LINE_DEMERITS + badness(ratio)).pow(2);

    if item.penalty() >= 0 {
        d = d.saturating_add(item.penalty().pow(2) as u32);
    } else if item.penalty() != -INFINITE_PENALTY {
        d = d.saturating_sub(item.penalty().pow(2) as u32);
    }

    if item.flagged() && from_item.flagged() {
        d = d.saturating_add(FLAGGED_DEMERITS);
    }

    if (class as i32 - active.fitness_class as i32).abs() > 1 {
        d = d.saturating_add(FITNESS_DEMERITS);
    }

    d = d.saturating_add(active.demerits);

    d
}

#[inline]
fn sums_after<T>(sums: &Sums, items: &[Item<T>], position: usize) -> Sums {
    let mut sums = *sums;

    for i in position..items.len() {
        match items[i] {
            Item::Box { .. } => break,
            Item::Glue { width, stretch, shrink } => {
                sums.width += width;
                sums.stretch += stretch;
                sums.shrink += shrink;
            },
            Item::Penalty { penalty, .. } if penalty == -INFINITE_PENALTY && i > position => break,
            _ => {},
        }
    }

    sums
}

#[inline]
fn explore<T>(nodes: &mut Vec<Node>, head: &mut usize, items: &[Item<T>], lengths: &[i32], sums: &Sums, threshold: f32, boundary: usize, position: usize) {
    let mut current = *head;
    let mut previous = NULL;

    loop {
        let mut min_demerits = ::std::u32::MAX;
        let mut candidates = [Candidate { demerits: ::std::u32::MAX,
                                          address: NULL,
                                          ratio: 0.0,
                                          width: 0 }; 4];
        loop {
            let next = nodes[current].next;
            let line = nodes[current].line + 1;
            let ideal_len = lengths[line.min(lengths.len() - 1)];
            let (ratio, actual_len) = ratio(ideal_len, sums, &nodes[current].sums, &items[position]);

            if ratio < -1.0 || items[position].penalty() == -INFINITE_PENALTY {
                // Deactivate node.
                if previous != NULL {
                    nodes[previous].next = next;
                } else {
                    *head = next;
                }
            } else {
                previous = current;
            }

            if ratio >= -1.0 && ratio <= threshold {
                let class = fitness_class(ratio);
                let d = demerits(ratio, class, &nodes[current],
                                 &items[position], &items[nodes[current].position]);
                if d < candidates[class].demerits {
                    candidates[class].demerits = d;
                    candidates[class].address = current;
                    candidates[class].ratio = ratio;
                    candidates[class].width = actual_len;
                    if d < min_demerits {
                        min_demerits = d;
                    }
                }
            }

            current = next;

            if current == NULL {
                break;
            }

            if nodes[current].line >= line && line < boundary {
                break;
            }
        }

        if min_demerits < ::std::u32::MAX {
            for c in 0..candidates.len() {
                if candidates[c].demerits < min_demerits + FITNESS_DEMERITS {
                    let sums_after = sums_after(sums, items, position);
                    // Activate node.
                    let new_addr = nodes.len();
                    let mut node = Node {
                        position,
                        line: nodes[candidates[c].address].line + 1,
                        fitness_class: c,
                        sums: sums_after,
                        ratio: candidates[c].ratio,
                        width: candidates[c].width,
                        demerits: candidates[c].demerits,
                        best_from: candidates[c].address,
                        next: current,
                    };
                    if previous != NULL {
                        nodes[previous].next = new_addr;
                    } else {
                        *head = new_addr;
                    }
                    previous = new_addr;
                    nodes.push(node);
                }
            }
        }

        if current == NULL {
            break;
        }
    }
}

pub fn total_fit<T>(items: &[Item<T>], lengths: &[i32], mut threshold: f32, looseness: i32) -> Vec<Breakpoint> {
    let boundary = if looseness != 0 {
        ::std::usize::MAX
    } else {
        lengths.len().saturating_sub(2)
    };

    // Avoid overflows in the demerits computation.
    threshold = threshold.min(8.6);

    let mut nodes = Vec::with_capacity(items.len());
    nodes.push(Node::default());

    let mut head = 0;
    let mut sums = Sums { width: 0, stretch: 0, shrink: 0 };

    for position in 0..items.len() {
        match items[position] {
            Item::Box { width, .. } => sums.width += width,
            Item::Glue { width, stretch, shrink } => {
                if position > 0 && items[position-1].is_box() {
                    explore(&mut nodes, &mut head, items, lengths, &sums, threshold, boundary, position);
                }
                sums.width += width;
                sums.stretch += stretch;
                sums.shrink += shrink;
            },
            Item::Penalty { penalty, .. } if penalty != INFINITE_PENALTY => {
                explore(&mut nodes, &mut head, items, lengths, &sums, threshold, boundary, position);
            },
            _ => {},
        }

        if head == NULL {
            break;
        }
    }

    if head == NULL {
        return vec![];
    }

    let mut current = head;
    let mut chosen = NULL;
    let mut d = ::std::u32::MAX;

    while current != NULL {
        if nodes[current].demerits < d {
            d = nodes[current].demerits;
            chosen = current;
        }
        current = nodes[current].next;
    }

    let line = nodes[chosen].line;

    if looseness != 0 {
        let mut current = head;
        let mut drift = 0;

        while current != NULL {
            let delta = nodes[current].line as i32 - line as i32;
            if (looseness <= delta && delta < drift) || (drift < delta && delta <= looseness) {
                drift = delta;
                d = nodes[current].demerits;
                chosen = current;
            } else if delta == drift && nodes[current].demerits < d {
                d = nodes[current].demerits;
                chosen = current;
            }
            current = nodes[current].next;
        }
    }

    let mut result = Vec::new();

    while chosen != NULL {
        let node = nodes[chosen];
        result.push(Breakpoint { position: node.position,
                                 ratio: node.ratio,
                                 width: node.width });
        chosen = node.best_from;
    }

    result.pop();
    result.reverse();
    result
}

pub fn standard_fit<T>(items: &[Item<T>], lengths: &[i32], threshold: f32) -> Vec<Breakpoint> {
    let mut result = Vec::new();
    let mut position = 0;
    let mut previous_position = position;
    let mut line = 0;
    let mut ideal_len = lengths[line.min(lengths.len() - 1)];
    let mut sums = Sums::default();
    let mut previous_sums = sums;
    let mut current;

    while position < items.len() {
        current = &items[position];

        match current {
            Item::Box { width, .. } => sums.width += width,
            Item::Glue { width, stretch, shrink } => {
                if (sums.width - previous_sums.width) > ideal_len && position > previous_position && items[position - 1].is_box() {
                    let (mut r, mut w) = ratio(ideal_len, &sums, &previous_sums, current);

                    if r < -1.0 {
                        let high_position = position;
                        let high_sums = sums;
                        position -= 1;

                        while position > previous_position {
                            current = &items[position];

                            match current {
                                Item::Box { width, .. } => {
                                    sums.width -= width;
                                }
                                Item::Glue { width, stretch, shrink } => {
                                    sums.width -= width;
                                    sums.stretch -= stretch;
                                    sums.shrink -= shrink;
                                    if items[position - 1].is_box() {
                                        break;
                                    }
                                },
                                _ => (),
                            }

                            position -= 1;
                        }

                        if position == previous_position {
                            return vec![];
                        }

                        let (r1, w1) = ratio(ideal_len, &sums, &previous_sums, current);
                        r = r1; w = w1;

                        if r > threshold {
                            let low_position = position;
                            let low_sums = sums;
                            let low_ratio = r;
                            let low_width = w;
                            position = high_position;
                            sums = high_sums;

                            while position > low_position {
                                current = &items[position];

                                match current {
                                    Item::Box { width, .. } => sums.width -= width,
                                    Item::Penalty { width, .. } if *width > 0 => {
                                        let (r2, w2) = ratio(ideal_len, &sums, &previous_sums, current);
                                        r = r2; w = w2;
                                        if r >= -1.0 && r <= threshold {
                                            break;
                                        }
                                    },
                                    _ => (),
                                }

                                position -= 1;
                            }

                            if position == low_position {
                                sums = low_sums;
                                r = low_ratio;
                                w = low_width;
                            }
                        }
                    }

                    previous_position = position;
                    previous_sums = sums;
                    result.push(Breakpoint { position, ratio: r, width: w });

                    position += 1;

                    while position < items.len() {
                        current = &items[position];
                        if current.is_box() {
                            break;
                        }
                        position += 1;
                    }

                    line += 1;
                    ideal_len = lengths[line.min(lengths.len() - 1)];

                    continue;
                } else {
                    sums.width += width;
                    sums.stretch += stretch;
                    sums.shrink += shrink;
                }
            },
            Item::Penalty { penalty, .. } if *penalty == -INFINITE_PENALTY => {
                let (r, w) = ratio(ideal_len, &sums, &previous_sums, current);
                result.push(Breakpoint { position, ratio: r, width: w });
                previous_position = position;
                previous_sums = sums;
                line += 1;
                ideal_len = lengths[line.min(lengths.len() - 1)];
            },
            _ => (),
        }

        position += 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    const HYPHEN_PENALTY: i32 = 50;

    // Excerpt from *The Frog Prince* by Brothers Grimm.
    // Optional breaks are marked with soft hyphens, the only explicit hyphen is in *lime-tree*.
    const FROG_PRINCE: &str = "In olden times when wish­ing still helped one, there lived a king whose daugh­ters were all beau­ti­ful; and the young­est was so beau­ti­ful that the sun it­self, which has seen so much, was aston­ished when­ever it shone in her face. Close by the king’s castle lay a great dark for­est, and un­der an old lime-tree in the for­est was a well, and when the day was very warm, the king’s child went out into the for­est and sat down by the side of the cool foun­tain; and when she was bored she took a golden ball, and threw it up on high and caught it; and this ball was her favor­ite play­thing.";

    // Traditional Monotype characters widths (in *machine units* (1/18th of an em)),
    // given by Donald Knuth in *Digital Typography*, p. 75.
    const LOWERCASE_WIDTHS: [i32; 26] = [9, 10, 8, 10, 8, 6, 9, 10, 5, 6, 10, 5, 15,
                                         10, 9, 10, 10, 7, 7, 7, 10, 9, 13, 10, 10, 8];
    fn char_width(c: char) -> i32 {
        match c {
            'a' ... 'z' => LOWERCASE_WIDTHS[c as usize - 'a' as usize],
            'C' => 13,
            'I' | '-' | '­' | ' ' => 6,
            ',' | ';' | '.' | '’' => 5,
            _ => 0,
        }
    }

    fn glue_after<T>(c: char) -> Item<T> {
        match c {
            ',' => Item::Glue { width: 6, stretch: 4, shrink: 2 },
            ';' => Item::Glue { width: 6, stretch: 4, shrink: 1 },
            '.' => Item::Glue { width: 8, stretch: 6, shrink: 1 },
            _ => Item::Glue { width: 6, stretch: 3, shrink: 2 },
        }
    }

    fn make_items(text: &str) -> Vec<Item<()>> {
        let mut result = Vec::new();
        let mut buf = String::new();
        let mut width = 0;
        let mut last_c = '*';

        for c in text.chars() {
            if "- ­".find(c).is_some() {
                if !buf.is_empty() {
                    result.push(Item::Box { width, data: () });
                    buf.clear();
                    width = 0;
                }
            }

            match c {
                ' ' => result.push(glue_after(last_c)),
                '-' => {
                    result.push(Item::Box { width: char_width(c), data: () });
                    result.push(Item::Penalty { width: 0,
                                                penalty: HYPHEN_PENALTY,
                                                flagged: true });
                },
                // Soft hyphen.
                '­' => result.push(Item::Penalty { width: char_width(c),
                                                   penalty: HYPHEN_PENALTY,
                                                   flagged: true }),
                _ => {
                    buf.push(c);
                    width += char_width(c);
                },
            }

            last_c = c;
        }

        if !buf.is_empty() {
            result.push(Item::Box { width, data: () });
        }

        result.extend_from_slice(&[Item::Penalty { penalty: INFINITE_PENALTY,  width: 0, flagged: false },
                                   Item::Glue { width: 0, stretch: INFINITE_PENALTY, shrink: 0 },
                                   Item::Penalty { penalty: -INFINITE_PENALTY,  width: 0, flagged: true }]);

        result
    }

    macro_rules! pos {
        ($x:expr) => ($x.iter().map(|x| x.position).collect::<Vec<usize>>());
    }

    #[test]
    fn test_breakpoints() {
        let mut items = make_items(FROG_PRINCE);
        items.insert(0, Item::Glue { width: 18, stretch: 0, shrink: 0 });
        let narrow = total_fit(&items, &[390], 1.0, 0);
        let medium = total_fit(&items, &[500], 1.0, 0);
        let medium_tight = total_fit(&items, &[500], 1.0, -1);
        let medium_loose = total_fit(&items, &[500], 2.5, 1);

        // Knuth, Donald: Digital Typography, p. 81.
        assert_eq!(pos!(narrow), vec![18, 38, 64, 84, 106, 130, 155, 175, 199, 221, 241, 263]);
        // Ibid, p. 113.
        assert_eq!(pos!(medium), vec![24, 52, 82, 108, 141, 169, 199, 225, 253, 263]);
        assert_eq!(pos!(medium_tight), vec![26, 54, 84, 112, 147, 173, 205, 233, 263]);
        assert_eq!(pos!(medium_loose), vec![22, 48, 78, 102, 130, 159, 183, 209, 235, 259, 263]);
        // If the algorithm can't satisfy the constraints, the return value is empty.
        let too_narrow = total_fit(&items, &[100], 1.0, 0);
        assert!(too_narrow.is_empty());

        // *Standard* algorithm (an informal description is given ibid, p. 68, last paragraph).
        let std_narrow = standard_fit(&items, &[390], 1.0);
        let std_medium = standard_fit(&items, &[500], 1.0);
        assert_eq!(pos!(std_narrow), vec![18, 40, 66, 86, 108, 132, 157, 177, 201, 221, 243, 263]);
        assert_eq!(pos!(std_medium), vec![26, 54, 84, 112, 147, 173, 205, 233, 263]);

        // If one of the boxes is larger than the line, the return value is empty.
        let absurd_items = vec![Item::Box { width: 100, data: () },
                                Item::Glue { width: 0, stretch: 0, shrink: 0 }];
        let std_absurd = standard_fit(&absurd_items, &[90], 1.0);
        assert!(std_absurd.is_empty());
    }
}

const QUESTIONS_DATA = [
  // 10th Grade NCERT Physics Questions
  {
    board: 'NCERT',
    grade: '10th Grade',
    topic: 'Electricity',
    difficulty: 'basic',
    type: 'mcq',
    question: 'What is the SI unit of electric current?',
    options: ['Volt', 'Ohm', 'Ampere', 'Watt'],
    answer: 'Ampere',
  },
  {
    board: 'NCERT',
    grade: '10th Grade',
    topic: 'Electricity',
    difficulty: 'moderate',
    type: 'fill_in_the_blanks',
    question: 'The resistance of a conductor is directly proportional to its ______.',
    answer: 'length',
  },
  {
    board: 'NCERT',
    grade: '10th Grade',
    topic: 'Electricity',
    difficulty: 'advanced',
    type: 'reasoning',
    question: 'Assertion: The magnetic field produced by a solenoid is independent of its radius. Reason: The magnetic field inside a solenoid is uniform.',
    options: [
      'Both A and R are true and R is the correct explanation of A.',
      'Both A and R are true but R is not the correct explanation of A.',
      'A is true but R is false.',
      'A is false but R is true.'
    ],
    answer: 'Both A and R are true and R is not the correct explanation of A.',
  },
  // Add more questions here for different topics and grades
];

export default QUESTIONS_DATA;
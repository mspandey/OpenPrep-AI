const express = require('express');
const { protect } = require('../middleware/auth');
const { universalSearch } = require('../controllers/searchController');

const router = express.Router();

router.get('/', protect, universalSearch);

module.exports = router;

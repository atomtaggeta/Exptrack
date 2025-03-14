document.addEventListener('DOMContentLoaded', function() {
    // Mobile menu toggle
    const mobileMenuButton = document.getElementById('mobileMenuButton');
    const navbarLinks = document.getElementById('navbarLinks');
    
    if (mobileMenuButton && navbarLinks) {
        mobileMenuButton.addEventListener('click', function() {
            navbarLinks.classList.toggle('show');
        });
    }
    
    // Close mobile menu when clicking outside
    document.addEventListener('click', function(event) {
        if (navbarLinks && navbarLinks.classList.contains('show') && 
            !event.target.closest('.navbar-container')) {
            navbarLinks.classList.remove('show');
        }
    });
    
    // Improved theme detection and application
    function applyTheme(theme) {
        const htmlElement = document.documentElement;
        
        if (theme === 'dark') {
            htmlElement.classList.remove('light');
            htmlElement.classList.add('dark');
        } else if (theme === 'light') {
            htmlElement.classList.remove('dark');
            htmlElement.classList.add('light');
        } else if (theme === 'system') {
            // Check system preference
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            if (prefersDark) {
                htmlElement.classList.remove('light');
                htmlElement.classList.add('dark');
            } else {
                htmlElement.classList.remove('dark');
                htmlElement.classList.add('light');
            }
        }
    }
    
    // Get theme from local storage or use default 'light'
    const storedTheme = localStorage.getItem('theme') || 'light';
    
    // Apply the stored theme on all pages
    applyTheme(storedTheme);
    
    // Theme switcher functionality on settings page - REMOVED IMMEDIATE CHANGE
    const themeSelect = document.getElementById('theme');
    if (themeSelect) {
        // Set the select value to match the current theme
        themeSelect.value = storedTheme;
        
        // REMOVED: The event listener that immediately changes theme
        // Now theme will only change after form submission
    }
    
    // Listen for system theme changes if using system setting
    const systemThemeMedia = window.matchMedia('(prefers-color-scheme: dark)');
    systemThemeMedia.addEventListener('change', function() {
        if (localStorage.getItem('theme') === 'system') {
            applyTheme('system');
        }
    });
    
    // Automatic category icon selection based on expense description
    const expenseInput = document.getElementById('content');
    const categorySelect = document.getElementById('category');
    
    if (expenseInput && categorySelect) {
        expenseInput.addEventListener('blur', function() {
            const text = this.value.toLowerCase();
            
            // Simple keyword matching for auto-categorization
            if (/restaurant|food|grocery|eat|dinner|lunch|breakfast|cafe|coffee|pizza|burger|takeout/.test(text)) {
                categorySelect.value = 'Food';
            } else if (/uber|lyft|taxi|car|gas|fuel|train|bus|subway|metro|transport|travel|flight|airplane/.test(text)) {
                categorySelect.value = 'Transportation';
            } else if (/movie|netflix|hulu|disney|spotify|concert|show|theater|game|entertainment/.test(text)) {
                categorySelect.value = 'Entertainment';
            } else if (/rent|mortgage|electricity|water|gas|internet|cable|bill|utility/.test(text)) {
                categorySelect.value = 'Bills';
            }
        });
    }
    
    // Form validation
    const forms = document.querySelectorAll('form:not([action="/logout"]):not(.settings-content form)');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const amountInput = this.querySelector('input[type="number"]');
            if (amountInput && parseFloat(amountInput.value) < 0) {
                e.preventDefault();
                alert('Amount cannot be negative');
            }
        });
    });
    
    // Make tables sortable
    const tables = document.querySelectorAll('table');
    tables.forEach(table => {
        const headers = table.querySelectorAll('th');
        headers.forEach((header, index) => {
            header.style.cursor = 'pointer';
            header.addEventListener('click', function() {
                sortTable(table, index);
            });
        });
    });
    
    // Setup custom confirmation modals
    setupDeleteConfirmation();
    setupLogoutConfirmation();
    
    // Apply category colors to expense items
    applyCategoryColors();
});
// Function to sort tables
function sortTable(table, columnIndex) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const isNumeric = table.rows[0].cells[columnIndex].textContent.trim().startsWith('$');
    const direction = table.getAttribute('data-sort-direction') === 'asc' ? -1 : 1;
    
    rows.sort((a, b) => {
        let aValue = a.cells[columnIndex].textContent.trim();
        let bValue = b.cells[columnIndex].textContent.trim();
        
        if (isNumeric) {
            aValue = parseFloat(aValue.replace(/[^0-9.-]+/g, ''));
            bValue = parseFloat(bValue.replace(/[^0-9.-]+/g, ''));
        }
        
        if (aValue < bValue) return -1 * direction;
        if (aValue > bValue) return 1 * direction;
        return 0;
    });
    
    // Remove existing rows
    while (tbody.firstChild) {
        tbody.removeChild(tbody.firstChild);
    }
    
    // Add sorted rows
    rows.forEach(row => {
        tbody.appendChild(row);
    });
    
    // Toggle direction for next click
    table.setAttribute('data-sort-direction', direction === 1 ? 'asc' : 'desc');
}

// Create a custom confirmation modal instead of using the browser's default
function createConfirmModal() {
    // Create modal elements if they don't exist yet
    if (!document.getElementById('customConfirmModal')) {
        const modalHTML = `
            <div id="customConfirmModal" class="confirm-modal">
                <div class="confirm-modal-content">
                    <div class="confirm-modal-header">
                        <h3><i class="fas fa-exclamation-triangle"></i> Confirm Deletion</h3>
                    </div>
                    <div class="confirm-modal-body">
                        <p>Are you sure you want to delete this expense?</p>
                    </div>
                    <div class="confirm-modal-footer">
                        <button id="confirmCancel" class="button button-secondary">Cancel</button>
                        <button id="confirmDelete" class="button button-danger">Delete</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.insertAdjacentHTML('beforeend', modalHTML);
    }
    
    const modal = document.getElementById('customConfirmModal');
    const cancelBtn = document.getElementById('confirmCancel');
    const deleteBtn = document.getElementById('confirmDelete');
    
    return { modal, cancelBtn, deleteBtn };
}

// Handle delete links with custom confirmation
function setupDeleteConfirmation() {
    // Get all delete links
    const deleteLinks = document.querySelectorAll('a[href^="/delete/"]');
    
    deleteLinks.forEach(link => {
        // Remove any existing event listeners
        const newLink = link.cloneNode(true);
        link.parentNode.replaceChild(newLink, link);
        
        newLink.addEventListener('click', function(e) {
            e.preventDefault();
            const deleteUrl = this.getAttribute('href');
            
            // Create and show the confirmation modal
            const { modal, cancelBtn, deleteBtn } = createConfirmModal();
            modal.classList.add('show');
            
            // Handle Cancel button
            cancelBtn.onclick = function() {
                modal.classList.remove('show');
            };
            
            // Handle Delete button
            deleteBtn.onclick = function() {
                window.location.href = deleteUrl;
            };
            
            // Also close when clicking outside the modal
            modal.addEventListener('click', function(event) {
                if (event.target === modal) {
                    modal.classList.remove('show');
                }
            });
        });
    });
}

// Create a custom logout confirmation modal
function createLogoutConfirmModal() {
    // Create modal elements if they don't exist yet
    if (!document.getElementById('logoutConfirmModal')) {
        const modalHTML = `
            <div id="logoutConfirmModal" class="confirm-modal">
                <div class="confirm-modal-content">
                    <div class="confirm-modal-header">
                        <h3><i class="fas fa-sign-out-alt"></i> Confirm Logout</h3>
                    </div>
                    <div class="confirm-modal-body">
                        <p>Are you sure you want to log out?</p>
                    </div>
                    <div class="confirm-modal-footer">
                        <button id="logoutCancel" class="button button-secondary">Cancel</button>
                        <button id="logoutConfirm" class="button button-primary">Logout</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.insertAdjacentHTML('beforeend', modalHTML);
    }
    
    const modal = document.getElementById('logoutConfirmModal');
    const cancelBtn = document.getElementById('logoutCancel');
    const confirmBtn = document.getElementById('logoutConfirm');
    
    return { modal, cancelBtn, confirmBtn };
}

// Function to handle logout confirmation
function setupLogoutConfirmation() {
    // Get logout form
    const logoutForm = document.querySelector('form[action="/logout"]');
    
    if (logoutForm) {
        logoutForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Create and show the confirmation modal
            const { modal, cancelBtn, confirmBtn } = createLogoutConfirmModal();
            modal.classList.add('show');
            
            // Handle Cancel button
            cancelBtn.onclick = function() {
                modal.classList.remove('show');
            };
            
            // Handle Confirm button
            confirmBtn.onclick = function() {
                logoutForm.submit();
            };
            
            // Close when clicking outside the modal
            modal.addEventListener('click', function(event) {
                if (event.target === modal) {
                    modal.classList.remove('show');
                }
            });
        });
    }
}

// Apply category-based colors to expense items
function applyCategoryColors() {
    // Get all expense items
    const expenseItems = document.querySelectorAll('.expense-item');
    
    expenseItems.forEach(item => {
        // Find the category badge
        const categoryBadge = item.querySelector('.category-badge');
        if (categoryBadge) {
            // Extract category text - trim whitespace and convert to lowercase
            const categoryText = categoryBadge.textContent.trim();
            const categoryClass = 'category-' + categoryText.toLowerCase().replace(/\//g, '-').replace(/\s+/g, '-');
            
            // Add category class to the expense item
            item.classList.add(categoryClass);
        }
    });
}
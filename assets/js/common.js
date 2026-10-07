$(document).ready(function() {
    $('a.abstract').click(function() {
        $(this).parent().parent().find(".abstract.hidden").toggleClass('open');
        $(this).parent().parent().find(".bibtex.hidden.open").toggleClass('open');
    });
    $('a.bibtex').click(function() {
        $(this).parent().parent().find(".bibtex.hidden").toggleClass('open');
        $(this).parent().parent().find(".abstract.hidden.open").toggleClass('open');
    });
    $('a').removeClass('waves-effect waves-light');

    function closePublicationPreview() {
        $('.publication-preview-modal').remove();
    }

    $('.publication-preview').on('click', function(event) {
        var media = $(this).find('img, video').first();
        if (!media.length) return;

        var modal = $('<div class="publication-preview-modal" role="dialog" aria-label="Publication preview"></div>');
        var expanded = media.clone().removeClass('preview').removeAttr('controls');
        expanded.attr('alt', media.attr('alt') || 'Publication preview');
        modal.append(expanded);
        $('body').append(modal);
        modal.on('click', closePublicationPreview);
        modal.on('keydown', function(event) {
            if (event.key === 'Escape') closePublicationPreview();
        }).trigger('focus');
    });

    $('.publication-preview').on('keydown', function(event) {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            $(this).trigger('click');
        }
    });

    $(document).on('keydown', function(event) {
        if (event.key === 'Escape') closePublicationPreview();
    });
});

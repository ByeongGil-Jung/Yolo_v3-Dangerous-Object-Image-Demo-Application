$(document).ready(function() {
    $("#view-btn-id").click(function() {
        let selected_img_filename = $("#sample-img-filenames-select-id option:selected").val();
        let view_img = null;

        $.ajax({
            type: "POST",
            url: "/view",
            data: {sample_img_filename: selected_img_filename},
            dataType: "JSON",
            success: function(data) {
               view_img = data["view_base64_encoded_img"];

                $("#show-img-id").attr("src", "data:image/png;base64," + view_img);
            },
            error: function(xhr, status, error) {
                console.log(error);
            }
        });
    });

    $("#inference-btn-id").click(function() {
        let selected_img_filename = $("#sample-img-filenames-select-id option:selected").val();
        let inference_img = null;
        let result_text = null;

        $("#result-text-id").val("Pending ...");

        $.ajax({
            type: "POST",
            url: "/inference",
            data: {sample_img_filename: selected_img_filename},
            dataType: "JSON",
            success: function(data) {
                inference_img = data["inference_base64_encoded_img"];
                result_text = data["result_text"];
                console.log(result_text);

                $("#show-img-id").attr("src", "data:image/png;base64," + inference_img);
                $("#result-text-id").val(result_text);
            },
            error: function(xhr, status, error) {
                console.log(error);
            }
        });
    });
});